open Ir

type index_key = Constant_index of int | Dynamic_index of int
type address_key = Root of int | Pointer_value of int | Element of address_key * index_key

type state = {
  substitutions : (int, value) Hashtbl.t;
  integer_constants : (int, int) Hashtbl.t;
  pointer_addresses : (int, address_key) Hashtbl.t;
  (* Value numbering for pure integer arithmetic, keyed structurally. *)
  pure_values : (string, value) Hashtbl.t;
  mutable cached_loads : (value * value) list;
}

let create_state () = {
  substitutions = Hashtbl.create 16;
  integer_constants = Hashtbl.create 16;
  pointer_addresses = Hashtbl.create 16;
  pure_values = Hashtbl.create 64;
  cached_loads = [];
}

let rec resolve state value =
  match Hashtbl.find_opt state.substitutions value.id with
  | None -> value
  | Some replacement when replacement.id = value.id -> replacement
  | Some replacement -> resolve state replacement

let address_of state pointer =
  match Hashtbl.find_opt state.pointer_addresses pointer.id with
  | Some address -> address
  | None ->
      (match pointer.provenance with
       | Some root when pointer.id = root -> Root root
       | _ -> Pointer_value pointer.id)

let index_of state index =
  match Hashtbl.find_opt state.integer_constants index.id with
  | Some n -> Constant_index n
  | None -> Dynamic_index index.id

let same_value_facts a b = { a with id = b.id; loc = b.loc } = b

(* Integer index arithmetic is pure and deterministic, so two instructions with
   the same operator and operands compute the same value. Expansions the
   compiler generates are full of such repeats: an accumulator-position query
   re-derives the lane from the thread id at every call site, and without this
   each copy survives into PTX. Operands are already resolved to their
   representatives, so comparing ids is enough. Commutative operators sort
   their operands so the two orders share a key. *)
let pure_key state instruction =
  let id value = (resolve state value).id in
  let pair a b = (id a, id b) in
  let sorted a b = let x, y = pair a b in if x <= y then (x, y) else (y, x) in
  let binary name (x, y) = Some (Printf.sprintf "%s(%d,%d)" name x y) in
  match instruction with
  | Const_i32 (_, n) -> Some (Printf.sprintf "const(%d)" n)
  | Thread_idx_x _ -> Some "tid.x"
  | Global_idx_x _ -> Some "gid.x"
  | Block_idx_x _ -> Some "ctaid.x"
  | Block_idx_y _ -> Some "ctaid.y"
  | Add_i32 (_, a, b) -> binary "add" (sorted a b)
  | Mul_i32 (_, a, b) -> binary "mul" (sorted a b)
  | Sub_i32 (_, a, b) -> binary "sub" (pair a b)
  | Div_i32 (_, a, b) -> binary "div" (pair a b)
  | Min_i32 (_, a, b) -> binary "min" (sorted a b)
  | Rem_i32 (_, a, b) -> binary "rem" (pair a b)
  | _ -> None

(* [Some ()] when the instruction was replaced by an earlier equal one. *)
let reuse_pure state instruction =
  match pure_key state instruction, results instruction with
  | Some key, [dst] -> (
      match Hashtbl.find_opt state.pure_values key with
      | Some existing when same_value_facts dst existing ->
          Hashtbl.replace state.substitutions dst.id existing;
          Some ()
      | _ ->
          Hashtbl.replace state.pure_values key dst;
          None)
  | _ -> None

let clear_pure_values state = Hashtbl.reset state.pure_values

let find_cached_load state pointer result =
  let address = address_of state pointer in
  List.find_opt (fun (cached_pointer, cached_result) ->
    address_of state cached_pointer = address &&
    same_value_facts result cached_result)
    state.cached_loads

let invalidate_aliasing_stores kernel state written_pointer =
  state.cached_loads <- List.filter (fun (loaded_pointer, _) ->
    not (Alias_analysis.may_alias kernel loaded_pointer written_pointer))
    state.cached_loads

let clear_load_cache state = state.cached_loads <- []

(* Reuse a prior scalar load only while no intervening write may alias it.
   Uniqueness proves that a store through a distinct unique formal root cannot
   invalidate the cached value; aliased or unknown roots conservatively do. *)
let eliminate_redundant_loads kernel =
  Verifier.verify_exn kernel;
  let rec optimize_region body yield =
    let state = create_state () in
    let body = List.filter_map (fun instruction ->
      let instruction = map_uses (resolve state) instruction in
      match reuse_pure state instruction with
      | Some () -> None
      | None ->
      match instruction with
      | Const_i32 (dst, n) ->
          Hashtbl.replace state.integer_constants dst.id n;
          Some instruction
      | Gep_f32 (dst, base, index) ->
          let address = Element (address_of state base, index_of state index) in
          Hashtbl.replace state.pointer_addresses dst.id address;
          Some instruction
      | Load_f32 (dst, pointer) ->
          (match find_cached_load state pointer dst with
           | Some (_, cached_value) ->
               Hashtbl.replace state.substitutions dst.id cached_value;
               None
           | None ->
               state.cached_loads <- (pointer, dst) :: state.cached_loads;
               Some instruction)
      | Store_f32 (pointer, _) | Store_f32_masked (pointer, _, _, _) |
        Store_f32x2 (pointer, _, _) | Store_f32x4 (pointer, _, _, _, _) |
        Store_f32_grid_leader (pointer, _) | Store_tensor (pointer, _) |
        Store_tensor_masked (pointer, _, _, _) ->
          invalidate_aliasing_stores kernel state pointer;
          Some instruction
      | If (dst, condition, yes, no) ->
          clear_load_cache state;
          let optimize_branch (region : Ir.region) =
            let body, yield = optimize_region region.body region.yield in
            { body; yield }
          in
          Some (If (dst, condition, optimize_branch yes, optimize_branch no))
      | If_uni (dst, condition, yes, no) ->
          clear_load_cache state;
          let optimize_branch (region : Ir.region) =
            let body, yield = optimize_region region.body region.yield in
            { body; yield }
          in
          Some (If_uni (dst, condition, optimize_branch yes, optimize_branch no))
      | For (induction, limit, step, loop_body) ->
          clear_load_cache state;
          clear_pure_values state;
          let loop_body, _ = optimize_region loop_body None in
          Some (For (induction, limit, step, loop_body))
      | Barrier _ ->
          clear_load_cache state;
          Some instruction
      | _ -> Some instruction)
      body
    in
    body, Option.map (resolve state) yield
  in
  let body, _ = optimize_region kernel.body None in
  let optimized = { kernel with body } in
  Verifier.verify_exn optimized;
  optimized

let optimize = eliminate_redundant_loads
