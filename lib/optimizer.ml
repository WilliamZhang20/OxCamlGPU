open Ir

type index_key = Constant_index of int | Dynamic_index of int
type address_key = Root of int | Pointer_value of int | Element of address_key * index_key

let map_uses replace = function
  | (Const_i32 _ | Const_f32 _ | Thread_idx_x _ | Global_idx_x _ |
     Shared_alloc _ | Return None) as i -> i
  | Add_i32 (dst, a, b) -> Add_i32 (dst, replace a, replace b)
  | Add_f32 (dst, a, b) -> Add_f32 (dst, replace a, replace b)
  | Mul_f32 (dst, a, b) -> Mul_f32 (dst, replace a, replace b)
  | Gep_f32 (dst, base, index) -> Gep_f32 (dst, replace base, replace index)
  | Load_f32 (dst, ptr) -> Load_f32 (dst, replace ptr)
  | Load_f32_masked (dst, ptr, index, bound) ->
      Load_f32_masked (dst, replace ptr, replace index, replace bound)
  | Store_f32 (ptr, value) -> Store_f32 (replace ptr, replace value)
  | Store_f32_masked (ptr, value, index, bound) ->
      Store_f32_masked (replace ptr, replace value, replace index, replace bound)
  | Load_tensor (dst, src) -> Load_tensor (dst, replace src)
  | Store_tensor (dst, src) -> Store_tensor (replace dst, replace src)
  | Scale_tensor_f32 (dst, src, scalar) ->
      Scale_tensor_f32 (dst, replace src, replace scalar)
  | Mul_tensor_f32 (dst, a, b) -> Mul_tensor_f32 (dst, replace a, replace b)
  | Reduce_sum_f32 (dst, src) -> Reduce_sum_f32 (dst, replace src)
  | Tensor_lane_f32 (dst, src) -> Tensor_lane_f32 (dst, replace src)
  | Warp_reduce_sum_f32 (dst, src) -> Warp_reduce_sum_f32 (dst, replace src)
  | Store_f32_grid_leader (ptr, value) ->
      Store_f32_grid_leader (replace ptr, replace value)
  | Barrier _ as i -> i
  | Return (Some value) -> Return (Some (replace value))

let value_facts_equal_except_id a b = { a with id=b.id } = b

(* Reuse a prior scalar load only while no intervening write may alias it.
   Uniqueness proves that a store through a distinct unique formal root cannot
   invalidate the cached value; aliased or unknown roots conservatively do. *)
let eliminate_redundant_loads kernel =
  Verifier.verify_exn kernel;
  let substitutions = Hashtbl.create 16 in
  let integer_constants = Hashtbl.create 16 in
  let pointer_addresses = Hashtbl.create 16 in
  let rec resolve value = match Hashtbl.find_opt substitutions value.id with
    | None -> value
    | Some replacement when replacement.id = value.id -> replacement
    | Some replacement -> resolve replacement in
  let cached_loads = ref [] in
  let address_of pointer = match Hashtbl.find_opt pointer_addresses pointer.id with
    | Some key -> key
    | None -> (match pointer.provenance with
        | Some root -> Root root
        | None -> Pointer_value pointer.id) in
  let index_of index = match Hashtbl.find_opt integer_constants index.id with
    | Some n -> Constant_index n
    | None -> Dynamic_index index.id in
  let body = List.filter_map (fun instruction ->
    let instruction = map_uses resolve instruction in
    match instruction with
    | Const_i32 (dst, n) ->
        Hashtbl.replace integer_constants dst.id n;
        Some instruction
    | Gep_f32 (dst, base, index) ->
        Hashtbl.replace pointer_addresses dst.id (Element (address_of base, index_of index));
        Some instruction
    | Load_f32 (dst, ptr) ->
        (match List.find_opt (fun (cached_ptr, cached_value) ->
           address_of cached_ptr = address_of ptr &&
           value_facts_equal_except_id dst cached_value) !cached_loads with
         | Some (_, cached_value) ->
             Hashtbl.replace substitutions dst.id cached_value;
             None
         | None ->
             cached_loads := (ptr, dst) :: !cached_loads;
             Some instruction)
    | Store_f32 (ptr, _) | Store_f32_masked (ptr, _, _, _) |
      Store_f32_grid_leader (ptr, _) | Store_tensor (ptr, _) ->
        cached_loads := List.filter (fun (loaded_ptr, _) ->
          not (Verifier.may_alias kernel loaded_ptr ptr)) !cached_loads;
        Some instruction
    | Barrier _ ->
        cached_loads := [];
        Some instruction
    | _ -> Some instruction) kernel.body in
  let optimized = { kernel with body } in
  Verifier.verify_exn optimized;
  optimized

let optimize = eliminate_redundant_loads
