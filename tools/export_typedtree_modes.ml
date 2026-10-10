(* Walk the version-specific Typedtree and construct the shared Kernel_ast. *)
open Typedtree
module K = Kernel_ast
module G = Gpu_type
module T = Typedtree_support
open T

type binding =
  | Scalar of K.atom * (int64 * int64) option
  | Const_bool of K.atom * bool
  | Memory of K.memory
  | Shared of K.atom
  | Mbarrier of K.atom
  | Tensor_map of K.atom
  | Wgmma_acc of int list
  | Func of func
  | Void

and func = {
  params : (Types.arg_label * Ident.t) list;
  body : expression;
  env : (Ident.t * binding) list;
}

type state = { mutable next : int; mutable body : K.instruction list }

let fresh s = let n = s.next in s.next <- n + 1; n
let emit s exp op = s.body <- K.instruction ~loc:(span exp.exp_loc) op :: s.body
let full_range = Some (-2147483648L, 2147483647L)

let get env exp path = match path with
  | Path.Pident id ->
      (match List.find_opt (fun (key, _) -> Ident.same key id) env with
       | Some (_, v) -> v
       | None -> fail exp.exp_loc "unbound GPU value")
  | _ -> fail exp.exp_loc "unsupported nonlocal value"

let scalar loc = function
  | Scalar (a, r) -> a, r
  | Const_bool (a, _) -> a, None
  | _ -> fail loc "expected scalar"
let memory loc = function Memory m -> m | _ -> fail loc "expected GPU buffer"
let shared loc = function Shared a -> a | _ -> fail loc "expected CTA shared buffer"
let mbarrier loc = function Mbarrier a -> a | _ -> fail loc "expected mbarrier"
let tensor_map loc = function Tensor_map a -> a | _ -> fail loc "expected tensor_map"
let wgmma_acc loc = function Wgmma_acc ids -> ids | _ -> fail loc "expected wgmma_acc"
let unit loc = function Void -> () | _ -> fail loc "expected unit"

let checked loc (lo, hi) =
  if lo < -2147483648L || hi > 2147483647L then
    fail loc
      "native int arithmetic may overflow the supported i32 range; use explicit Int32 arithmetic"
  else Some (lo, hi)

(* Two's-complement wrap into signed i32. Products of i32 values fit in int64. *)
let i32_wrap n =
  let two32 = 4294967296L in
  let two31 = 2147483648L in
  let u = Int64.rem n two32 in
  let u = if u < 0L then Int64.add u two32 else u in
  if u >= two31 then Int64.sub u two32 else u

let const_extent loc v =
  match scalar loc v with
  | _, Some (lo, hi) when lo = hi && lo > 0L && lo <= Int64.of_int max_int ->
      Int64.to_int lo
  | _ -> fail loc "shared/descriptor extents must be positive compile-time constants"

let const_nat loc v =
  match scalar loc v with
  | _, Some (lo, hi) when lo = hi && lo >= 0L && lo <= Int64.of_int max_int ->
      Int64.to_int lo
  | _ -> fail loc "expected a non-negative compile-time constant"

let positional loc xs =
  List.map
    (function
      | Nolabel, Typedtree.Arg (e, _) -> e
      | _ -> fail loc "only positional fully applied calls are supported")
    xs

let labelled loc ~required xs =
  let tbl = Hashtbl.create 8 in
  let unit_arg = ref None in
  List.iter
    (function
      | Labelled name, Typedtree.Arg (e, _) -> Hashtbl.replace tbl name e
      | Nolabel, Typedtree.Arg (e, _) -> unit_arg := Some e
      | Optional _, _ -> fail loc "optional arguments unsupported"
      | _ -> fail loc "unsupported argument form")
    xs;
  let get name =
    match Hashtbl.find_opt tbl name with
    | Some e -> e
    | None -> fail loc ("missing labelled argument ~" ^ name)
  in
  List.map get required, !unit_arg

let bool_const s exp value =
  let d = fresh s in
  emit s exp (K.Const_bool (d, value));
  Const_bool (K.Value d, value)

let function_params loc params =
  List.map
    (fun (p : function_param) ->
      (match p.fp_arg_label with
       | Optional _ | Position _ -> fail loc "optional parameters unsupported"
       | Nolabel | Labelled _ -> ());
      (match p.fp_kind with
       | Tparam_pat { pat_desc = Tpat_var _ | Tpat_any; _ } -> ()
       | Tparam_pat { pat_desc = Tpat_construct (_, c, _, [], _); _ }
         when c.cstr_name = "()" -> ()
       | Tparam_pat pat ->
           fail pat.pat_loc "only simple function parameters are supported"
       | _ -> fail loc "optional parameters unsupported");
      p.fp_arg_label, p.fp_param)
    params

(* Inclusive trip count of a constant [for], or [None] when it must stay a loop.
   The cap keeps a large constant iteration from exploding the kernel. *)
let unroll_bounds = 64

let rec expression s env exp =
  let const op range =
    let d = fresh s in
    emit s exp (op d);
    Scalar (K.Value d, range)
  in
  let result op = const op None in
  match exp.exp_desc with
  | Texp_constant (Const_int n) ->
      ignore (checked exp.exp_loc (Int64.of_int n, Int64.of_int n));
      const (fun d -> K.Const_i32 (d, n)) (Some (Int64.of_int n, Int64.of_int n))
  | Texp_constant (Const_int32 n) ->
      const
        (fun d -> K.Const_i32 (d, Int32.to_int n))
        (Some (Int64.of_int32 n, Int64.of_int32 n))
  | Texp_constant (Const_int64 n) ->
      (* Used for gmma descriptor results typed as int64 / u64. *)
      if n < 0L then fail exp.exp_loc "negative int64 literal unsupported";
      result (fun d -> K.Const_i32 (d, Int64.to_int n))
  | Texp_constant (Const_float n) ->
      result (fun d -> K.Const_f32 (d, float_of_string n))
  | Texp_ident { path; _ } -> get env exp path
  | Texp_open (_, e) -> expression s env e
  | Texp_construct (_, c, _, [], _) ->
      (match kind exp.exp_env exp.exp_loc exp.exp_type, c.cstr_name with
       | Unit, "()" -> Void
      | Boolean, "true" -> bool_const s exp true
      | Boolean, "false" -> bool_const s exp false
       | _ -> fail exp.exp_loc "unsupported constructor")
  | Texp_let (Recursive, _, _) ->
      fail exp.exp_loc "recursive bindings are unsupported"
  | Texp_let (Nonrecursive, bindings, body) ->
      let rec bind env = function
        | [] -> expression s env body
        | binding :: rest ->
            (match binding.vb_pat.pat_desc, binding.vb_expr.exp_desc with
             | Tpat_tuple pats, Texp_apply (callee, xs, _, _, _, _) ->
                 let op = primitive callee in
                 (match op, List.map (function
                            | _, { pat_desc = Tpat_var { id; _ }; _ } -> id
                            | _, p -> fail p.pat_loc "f32x4 must bind four names") pats with
                  | ("load_f32x4" | "shared_load_f32x4"), [a; b; c; d] ->
                      let xs = positional binding.vb_expr.exp_loc xs in
                      let emit_load () =
                        match op, xs with
                        | "load_f32x4", [buffer; index] ->
                            let base = memory buffer.exp_loc (expression s env buffer) in
                            let ix, _ = scalar index.exp_loc (expression s env index) in
                            let p = fresh s in
                            emit s binding.vb_expr (K.Gep_f32 (p, base, ix));
                            let ia, ib, ic, id_ = fresh s, fresh s, fresh s, fresh s in
                            emit s binding.vb_expr
                              (K.Load_f32x4 (ia, ib, ic, id_, K.Value p));
                            ia, ib, ic, id_
                        | "shared_load_f32x4", [smem; index] ->
                            let m = shared smem.exp_loc (expression s env smem) in
                            let ix, _ = scalar index.exp_loc (expression s env index) in
                            let ia, ib, ic, id_ = fresh s, fresh s, fresh s, fresh s in
                            emit s binding.vb_expr
                              (K.Shared_load_f32x4 (ia, ib, ic, id_, m, ix));
                            ia, ib, ic, id_
                        | _ -> fail binding.vb_expr.exp_loc "invalid f32x4 arity"
                      in
                      let ia, ib, ic, id_ = emit_load () in
                      let env =
                        (a, Scalar (K.Value ia, None))
                        :: (b, Scalar (K.Value ib, None))
                        :: (c, Scalar (K.Value ic, None))
                        :: (d, Scalar (K.Value id_, None))
                        :: env
                      in
                      bind env rest
                  | _ -> fail binding.vb_pat.pat_loc "only f32x4 tuple bindings are supported")
             | _ ->
                 let value = expression s env binding.vb_expr in
                 (match binding.vb_pat.pat_desc with
                  | Tpat_var { id; _ } -> bind ((id, value) :: env) rest
                  | Tpat_any -> bind env rest
                  | _ ->
                      fail binding.vb_pat.pat_loc
                        "only simple kernel let bindings are supported"))
      in
      bind env bindings
  | Texp_sequence (a, _, b) ->
      unit a.exp_loc (expression s env a);
      expression s env b
  | Texp_ifthenelse (c, a, b) ->
      let cond = expression s env c in
      let else_branch () = Option.fold ~none:Void ~some:(expression s env) b in
      (match cond with
       | Const_bool (_, true) -> expression s env a
       | Const_bool (_, false) -> else_branch ()
       | _ ->
           let c, _ = scalar c.exp_loc cond in
           conditional s exp c (fun () -> expression s env a) else_branch)
  | Texp_function { params; body = Tfunction_body fbody; _ } ->
      Func { params = function_params exp.exp_loc params; body = fbody; env }
  | Texp_function _ ->
      fail exp.exp_loc "pattern-matching functions are unsupported"
  | Texp_apply (callee, xs, _, _, _, _) ->
      (match local_apply s env exp callee xs with
       | Some value -> value
       | None ->
      let op = primitive callee in
      (match op with
       | "shared" | "shared_wgmma" ->
           let layout = if op = "shared_wgmma" then `Wgmma else `Plain in
           (match labelled exp.exp_loc ~required:["rows"; "cols"] xs with
            | [rows; cols], unit_arg ->
                (match unit_arg with
                 | Some u -> unit u.exp_loc (expression s env u)
                 | None -> ());
                let rows = const_extent rows.exp_loc (expression s env rows) in
                let cols = const_extent cols.exp_loc (expression s env cols) in
                let d = fresh s in
                emit s exp (K.Shared_alloc (d, rows, cols, layout));
                Shared (K.Value d)
            | _ -> fail exp.exp_loc "shared expects ~rows and ~cols")
       | "wgmma_acc" ->
           (match labelled exp.exp_loc ~required:["n"] xs with
            | [n], unit_arg ->
                (match unit_arg with
                 | Some u -> unit u.exp_loc (expression s env u)
                 | None -> ());
                let n = const_extent n.exp_loc (expression s env n) in
                if n < 8 || n > 256 || n mod 8 <> 0 then
                  fail exp.exp_loc "wgmma_acc ~n must be in {8,16,...,256}";
                let acc_n = n / 2 in
                let ids =
                  List.init acc_n (fun _ ->
                    let d = fresh s in
                    emit s exp (K.Const_f32 (d, 0.));
                    d)
                in
                Wgmma_acc ids
            | _ -> fail exp.exp_loc "wgmma_acc expects ~n")
       | "gmma_descriptor" ->
           (match xs with
            | [ Nolabel, Typedtree.Arg (smem, _)
              ; Nolabel, Typedtree.Arg (index, _) ] ->
                let m = shared smem.exp_loc (expression s env smem) in
                let ix, _ = scalar index.exp_loc (expression s env index) in
                result (fun d -> K.Gmma_descriptor (d, m, ix))
            | _ -> fail exp.exp_loc "gmma_descriptor expects a shared tile and an element offset")
       | "wgmma_mma_tf32" ->
           let labs, _ = labelled exp.exp_loc ~required:["n"; "scale"] xs in
           let positionals =
             List.filter_map
               (function Nolabel, Typedtree.Arg (e, _) -> Some e | _ -> None)
               xs
           in
           (match positionals, labs with
            | [acc; da; db], [n; scale] ->
                let ids = wgmma_acc acc.exp_loc (expression s env acc) in
                let da, _ = scalar da.exp_loc (expression s env da) in
                let db, _ = scalar db.exp_loc (expression s env db) in
                let n = const_extent n.exp_loc (expression s env n) in
                let scale =
                  match scale.exp_desc with
                  | Texp_construct (_, c, _, [], _) ->
                      (match c.cstr_name with
                       | "true" -> true
                       | "false" -> false
                       | _ -> fail scale.exp_loc "scale must be true or false")
                  | _ -> fail scale.exp_loc "scale must be true or false"
                in
                emit s exp (K.Wgmma_mma_tf32 (ids, da, db, n, scale));
                Void
            | _ -> fail exp.exp_loc "wgmma_mma_tf32 expects acc, desc_a, desc_b, ~n, ~scale")
       (* Which warpgroup the calling thread is in. 128 is the warpgroup
          width, a hardware constant, so the division belongs here. *)
       | "warpgroup_index" ->
           (match xs with
            | [ Nolabel, Typedtree.Arg (u, _) ] ->
                unit u.exp_loc (expression s env u);
                let tid =
                  let d = fresh s in emit s exp (K.Thread_idx_x d); K.Value d
                in
                let width = let d = fresh s in emit s exp (K.Const_i32 (d, 128)); K.Value d in
                let d = fresh s in
                emit s exp (K.Div_i32 (d, tid, width));
                Scalar (K.Value d, full_range)
            | _ -> fail exp.exp_loc "warpgroup_index expects ()")
       (* Grouped CTA order. Row-major blockIdx makes every tile-row stream
          the whole of B, so B's DRAM traffic scales with the tile-row count.
          Walking [group] tile-rows before advancing along N keeps one A
          row-block resident across a group and cuts B's re-reads by that
          factor. This is the standard grouped bijection, so every tile is
          still covered exactly once, including when [tiles_m] is not a
          multiple of [group] and the last group is short. The group height is
          the author's tuning choice; the mapping is not. *)
       | ("grouped_tile_m" | "grouped_tile_n") ->
           (match labelled exp.exp_loc ~required:["group"; "tiles_m"; "tiles_n"] xs with
            | [group; tiles_m; tiles_n], unit_arg ->
                (match unit_arg with
                 | Some u -> unit u.exp_loc (expression s env u)
                 | None -> ());
                let value e = fst (scalar e.exp_loc (expression s env e)) in
                let group = value group
                and tiles_m = value tiles_m
                and tiles_n = value tiles_n in
                let bin node a b =
                  let d = fresh s in emit s exp (node d a b); K.Value d
                in
                let add a b = bin (fun d x y -> K.Add_i32 (d, x, y)) a b in
                let sub a b = bin (fun d x y -> K.Sub_i32 (d, x, y)) a b in
                let mul a b = bin (fun d x y -> K.Mul_i32 (d, x, y)) a b in
                let div a b = bin (fun d x y -> K.Div_i32 (d, x, y)) a b in
                let rem a b = bin (fun d x y -> K.Rem_i32 (d, x, y)) a b in
                let min_ a b = bin (fun d x y -> K.Min_i32 (d, x, y)) a b in
                let ctaid node =
                  let d = fresh s in emit s exp (node d); K.Value d
                in
                let bx = ctaid (fun d -> K.Block_idx_x d) in
                let by = ctaid (fun d -> K.Block_idx_y d) in
                (* Linear tile id, then the group it falls in. *)
                let pid = add (mul by tiles_n) bx in
                let per_group = mul group tiles_n in
                let group_first = mul (div pid per_group) group in
                let group_rows = min_ (sub tiles_m group_first) group in
                let value =
                  if op = "grouped_tile_m" then
                    add group_first (rem pid group_rows)
                  else div (rem pid per_group) group_rows
                in
                Scalar (value, full_range)
            | _ ->
                fail exp.exp_loc
                  "grouped_tile_m/n expect ~group ~tiles_m ~tiles_n")
       (* Where accumulator register [i] lives inside this warpgroup's
          m64 x N tile. wgmma.m64nNk8 puts lane [l] of warp [w] at
            row = w*16 + l/4 + 8*((i mod 4)/2)
            col = (l mod 4)*2 + (i mod 2) + 8*(i/4)
          which is hardware layout, not a schedule choice, so the compiler
          emits it rather than asking an author to rewrite it per kernel.
          Which 64 rows a warpgroup owns stays with the author. *)
       | ("wgmma_acc_row" | "wgmma_acc_col") ->
           let xs = positional exp.exp_loc xs in
           (match xs with
            | [acc; idx] ->
                let ids = wgmma_acc acc.exp_loc (expression s env acc) in
                (match expression s env idx with
                 | Scalar (_, Some (lo, hi)) when lo = hi && lo >= 0L &&
                     Int64.to_int lo < List.length ids ->
                     let i = Int64.to_int lo in
                     let konst n =
                       let d = fresh s in
                       emit s exp (K.Const_i32 (d, n)); K.Value d
                     in
                     let bin node a b =
                       let d = fresh s in emit s exp (node d a b); K.Value d
                     in
                     let rem a b = bin (fun d x y -> K.Rem_i32 (d, x, y)) a b in
                     let div a b = bin (fun d x y -> K.Div_i32 (d, x, y)) a b in
                     let mul a b = bin (fun d x y -> K.Mul_i32 (d, x, y)) a b in
                     let add a b = bin (fun d x y -> K.Add_i32 (d, x, y)) a b in
                     let tid =
                       let d = fresh s in emit s exp (K.Thread_idx_x d); K.Value d
                     in
                     let lane_wg = rem tid (konst 128) in
                     let lane = rem lane_wg (konst 32) in
                     let base, offset =
                       if op = "wgmma_acc_row" then
                         ( add (mul (div lane_wg (konst 32)) (konst 16))
                             (div lane (konst 4))
                         , 8 * (i mod 4 / 2) )
                       else
                         ( mul (rem lane (konst 4)) (konst 2)
                         , (i mod 2) + 8 * (i / 4) )
                     in
                     let value =
                       if offset = 0 then base else add base (konst offset)
                     in
                     Scalar (value, full_range)
                 | _ ->
                     fail idx.exp_loc
                       "accumulator position index must be a constant in range")
            | _ ->
                fail exp.exp_loc
                  "wgmma_acc_row/col expect acc and a constant index")
       | "wgmma_acc_get" ->
           let xs = positional exp.exp_loc xs in
           (match xs with
            | [acc; idx] ->
                let ids = wgmma_acc acc.exp_loc (expression s env acc) in
                (match expression s env idx with
                 | Scalar (a, Some (lo, hi)) when lo = hi && lo >= 0L &&
                     Int64.to_int lo < List.length ids ->
                     Scalar (K.Value (List.nth ids (Int64.to_int lo)), None)
                 | _ -> fail idx.exp_loc "wgmma_acc_get index must be a constant in range")
            | _ -> fail exp.exp_loc "wgmma_acc_get expects acc and constant index")
       | _ ->
           let xs = positional exp.exp_loc xs in
           (match op, xs with
            | ("&&" | "||"), [a; b] ->
                let a, _ = scalar a.exp_loc (expression s env a) in
                let literal value () = result (fun d -> K.Const_bool (d, value)) in
                if op = "&&" then
                  conditional s exp a (fun () -> expression s env b) (literal false)
                else
                  conditional s exp a (literal true) (fun () -> expression s env b)
            | "not", [a] ->
                let a, _ = scalar a.exp_loc (expression s env a) in
                conditional s exp a
                  (fun () -> result (fun d -> K.Const_bool (d, false)))
                  (fun () -> result (fun d -> K.Const_bool (d, true)))
            | ("thread_idx_x" | "global_idx_x" | "block_idx_x" | "block_idx_y"), [u] ->
                unit u.exp_loc (expression s env u);
                const
                  (fun d ->
                    match op with
                    | "thread_idx_x" -> K.Thread_idx_x d
                    | "global_idx_x" -> K.Global_idx_x d
                    | "block_idx_x" -> K.Block_idx_x d
                    | _ -> K.Block_idx_y d)
                  (Some (0L, 2147483647L))
            | ("load" | "load_masked"), buffer :: index :: tail ->
                let base = memory buffer.exp_loc (expression s env buffer) in
                let ix, _ = scalar index.exp_loc (expression s env index) in
                let bound =
                  match op, tail with
                  | "load", [] -> None
                  | "load_masked", [n] ->
                      Some (fst (scalar n.exp_loc (expression s env n)))
                  | _ -> fail exp.exp_loc "invalid load arity"
                in
                let p = fresh s in
                emit s exp (K.Gep_f32 (p, base, ix));
                result (fun d ->
                  match bound with
                  | None -> K.Load_f32 (d, K.Value p)
                  | Some n -> K.Load_f32_masked (d, K.Value p, ix, n))
            | ("store" | "store_masked"), buffer :: index :: tail ->
                let base = memory buffer.exp_loc (expression s env buffer) in
                let ix, _ = scalar index.exp_loc (expression s env index) in
                let bound, value =
                  match op, tail with
                  | "store", [v] -> None, v
                  | "store_masked", [n; v] ->
                      Some (fst (scalar n.exp_loc (expression s env n))), v
                  | _ -> fail exp.exp_loc "invalid store arity"
                in
                let p = fresh s in
                emit s exp (K.Gep_f32 (p, base, ix));
                let v, _ = scalar value.exp_loc (expression s env value) in
                emit s exp
                  (match bound with
                   | None -> K.Store_f32 (K.Value p, v)
                   | Some n -> K.Store_f32_masked (K.Value p, v, ix, n));
                Void
            | "store_f32x4", [buffer; index; a; b; c; d] ->
                let base = memory buffer.exp_loc (expression s env buffer) in
                let ix, _ = scalar index.exp_loc (expression s env index) in
                let p = fresh s in
                emit s exp (K.Gep_f32 (p, base, ix));
                let a, _ = scalar a.exp_loc (expression s env a) in
                let b, _ = scalar b.exp_loc (expression s env b) in
                let c, _ = scalar c.exp_loc (expression s env c) in
                let d, _ = scalar d.exp_loc (expression s env d) in
                emit s exp (K.Store_f32x4 (K.Value p, a, b, c, d));
                Void
            | "shared_load", [smem; index] ->
                let m = shared smem.exp_loc (expression s env smem) in
                let ix, _ = scalar index.exp_loc (expression s env index) in
                result (fun d -> K.Shared_load_f32 (d, m, ix))
            | "shared_store", [smem; index; v] ->
                let m = shared smem.exp_loc (expression s env smem) in
                let ix, _ = scalar index.exp_loc (expression s env index) in
                let v, _ = scalar v.exp_loc (expression s env v) in
                emit s exp (K.Shared_store_f32 (m, v, ix));
                Void
            | "shared_store_f32x4", [smem; index; a; b; c; d] ->
                let m = shared smem.exp_loc (expression s env smem) in
                let ix, _ = scalar index.exp_loc (expression s env index) in
                let a, _ = scalar a.exp_loc (expression s env a) in
                let b, _ = scalar b.exp_loc (expression s env b) in
                let c, _ = scalar c.exp_loc (expression s env c) in
                let d, _ = scalar d.exp_loc (expression s env d) in
                emit s exp (K.Shared_store_f32x4 (m, a, b, c, d, ix));
                Void
            | "barrier_cta", [u] ->
                unit u.exp_loc (expression s env u);
                emit s exp K.Barrier_cta;
                Void
            | "cp_async_f32x4", [smem; sidx; buffer; gidx] ->
                let m = shared smem.exp_loc (expression s env smem) in
                let si, _ = scalar sidx.exp_loc (expression s env sidx) in
                let g = memory buffer.exp_loc (expression s env buffer) in
                let gi, _ = scalar gidx.exp_loc (expression s env gidx) in
                emit s exp (K.Cp_async_f32x4 (m, si, g, gi));
                Void
            | "cp_async_commit", [u] ->
                unit u.exp_loc (expression s env u);
                emit s exp K.Cp_async_commit;
                Void
            | "cp_async_wait", [u] ->
                unit u.exp_loc (expression s env u);
                emit s exp K.Cp_async_wait_all;
                Void
            | "mad_f32", [acc; a; b] ->
                let acc, _ = scalar acc.exp_loc (expression s env acc) in
                let a, _ = scalar a.exp_loc (expression s env a) in
                let b, _ = scalar b.exp_loc (expression s env b) in
                result (fun d -> K.Mad_f32 (d, acc, a, b))
            | "store_grid_leader", [buffer; v] ->
                let p = memory buffer.exp_loc (expression s env buffer) in
                let v, _ = scalar v.exp_loc (expression s env v) in
                emit s exp (K.Store_grid_leader_f32 (p, v));
                Void
            | "warp_sum_f32", [v] ->
                let v, _ = scalar v.exp_loc (expression s env v) in
                result (fun d -> K.Warp_reduce_sum_f32 (d, v))
            | "mbarrier", [u] ->
                unit u.exp_loc (expression s env u);
                let d = fresh s in
                emit s exp (K.Mbarrier_alloc d);
                Mbarrier (K.Value d)
            | "mbarrier_set", [count] ->
                let count = const_extent count.exp_loc (expression s env count) in
                let d = fresh s in
                emit s exp (K.Mbarrier_set_alloc (d, count));
                Mbarrier (K.Value d)
            | "mbarrier_slot", [set; index] ->
                let set = mbarrier set.exp_loc (expression s env set) in
                let ix, _ = scalar index.exp_loc (expression s env index) in
                let d = fresh s in
                emit s exp (K.Mbarrier_slot (d, set, ix));
                Mbarrier (K.Value d)
            | "mbarrier_init", [m; c] ->
                let m = mbarrier m.exp_loc (expression s env m) in
                let c, _ = scalar c.exp_loc (expression s env c) in
                emit s exp (K.Mbarrier_init (m, c, None));
                Void
            | "mbarrier_init_elect", [m; c; p] ->
                let m = mbarrier m.exp_loc (expression s env m) in
                let c, _ = scalar c.exp_loc (expression s env c) in
                let p, _ = scalar p.exp_loc (expression s env p) in
                emit s exp (K.Mbarrier_init (m, c, Some p));
                Void
            | "mbarrier_arrive_expect_tx", [m; tx] ->
                let m = mbarrier m.exp_loc (expression s env m) in
                let tx, _ = scalar tx.exp_loc (expression s env tx) in
                emit s exp (K.Mbarrier_arrive_expect_tx (m, tx, None));
                Void
            | "mbarrier_arrive_expect_tx_elect", [m; tx; p] ->
                let m = mbarrier m.exp_loc (expression s env m) in
                let tx, _ = scalar tx.exp_loc (expression s env tx) in
                let p, _ = scalar p.exp_loc (expression s env p) in
                emit s exp (K.Mbarrier_arrive_expect_tx (m, tx, Some p));
                Void
            | "mbarrier_arrive", [m] ->
                let m = mbarrier m.exp_loc (expression s env m) in
                emit s exp (K.Mbarrier_arrive (m, None));
                Void
            | "mbarrier_arrive_elect", [m; p] ->
                let m = mbarrier m.exp_loc (expression s env m) in
                let p, _ = scalar p.exp_loc (expression s env p) in
                emit s exp (K.Mbarrier_arrive (m, Some p));
                Void
            | "mbarrier_try_wait_parity", [m; parity] ->
                let m = mbarrier m.exp_loc (expression s env m) in
                let parity, _ = scalar parity.exp_loc (expression s env parity) in
                emit s exp (K.Mbarrier_try_wait_parity (m, parity));
                Void
            | "tma_load_2d", [sm; idx; tmap; c0; c1; mbar] ->
                let sm = shared sm.exp_loc (expression s env sm) in
                let idx, _ = scalar idx.exp_loc (expression s env idx) in
                let tmap = tensor_map tmap.exp_loc (expression s env tmap) in
                let c0, _ = scalar c0.exp_loc (expression s env c0) in
                let c1, _ = scalar c1.exp_loc (expression s env c1) in
                let mbar = mbarrier mbar.exp_loc (expression s env mbar) in
                emit s exp (K.Tma_load_2d (sm, idx, tmap, c0, c1, mbar, None));
                Void
            | "tma_load_2d_elect", [sm; idx; tmap; c0; c1; mbar; p] ->
                let sm = shared sm.exp_loc (expression s env sm) in
                let idx, _ = scalar idx.exp_loc (expression s env idx) in
                let tmap = tensor_map tmap.exp_loc (expression s env tmap) in
                let c0, _ = scalar c0.exp_loc (expression s env c0) in
                let c1, _ = scalar c1.exp_loc (expression s env c1) in
                let mbar = mbarrier mbar.exp_loc (expression s env mbar) in
                let p, _ = scalar p.exp_loc (expression s env p) in
                emit s exp (K.Tma_load_2d (sm, idx, tmap, c0, c1, mbar, Some p));
                Void
            | "fence_proxy_async", [u] ->
                unit u.exp_loc (expression s env u);
                emit s exp K.Fence_proxy_async;
                Void
            | "wgmma_fence", [u] ->
                unit u.exp_loc (expression s env u);
                emit s exp K.Wgmma_fence;
                Void
            | "wgmma_commit_group", [u] ->
                unit u.exp_loc (expression s env u);
                emit s exp K.Wgmma_commit_group;
                Void
            | "wgmma_wait_group", [n] ->
                let n = const_nat n.exp_loc (expression s env n) in
                emit s exp (K.Wgmma_wait_group n);
                Void
            | ("i32_of_int" | "i32_to_int"), [v] -> expression s env v
            | ( "+." | "*." | "+" | "-" | "*" | "/" | "i32_add" | "i32_sub" | "i32_mul"
              | "i32_div" | "i32_rem" | "=" | "<>" | "<" | "<=" | ">" | ">=" )
              , [a; b] ->
                let av, ar = scalar a.exp_loc (expression s env a) in
                let bv, br = scalar b.exp_loc (expression s env b) in
                let singleton_int = function
                  | Some (lo, hi) when lo = hi -> Some lo
                  | _ -> None
                in
                let folded_int =
                  let div_overflow a b =
                    a = -2147483648L && b = -1L
                  in
                  match op, singleton_int ar, singleton_int br with
                  | ("/" | "i32_div"), Some a, Some b
                    when b <> 0L && not (div_overflow a b) ->
                      Some (Int64.div a b)
                  | "i32_rem", Some a, Some b
                    when b <> 0L && not (div_overflow a b) ->
                      Some (Int64.rem a b)
                  | ("+" | "-" | "*" | "i32_add" | "i32_sub" | "i32_mul"), Some a, Some b ->
                      Some
                        (match op with
                         | "+" | "i32_add" -> Int64.add a b
                         | "-" | "i32_sub" -> Int64.sub a b
                         | _ -> Int64.mul a b)
                  | _ -> None
                in
                (match folded_int with
                 | Some v ->
                     let v, range =
                       if String.starts_with ~prefix:"i32_" op then
                         let v = i32_wrap v in
                         v, Some (v, v)
                       else v, checked exp.exp_loc (v, v)
                     in
                     const (fun d -> K.Const_i32 (d, Int64.to_int v)) range
                 | None when op = "/" ->
                     fail exp.exp_loc
                       "native int division is only supported for non-zero compile-time constants"
                 | None ->
                (match op with
                 | "+." -> result (fun d -> K.Add_f32 (d, av, bv))
                 | "*." -> result (fun d -> K.Mul_f32 (d, av, bv))
                 | "+" | "-" | "*" | "i32_add" | "i32_sub" | "i32_mul" | "i32_div"
                 | "i32_rem" ->
                     let range =
                       if String.starts_with ~prefix:"i32_" op then full_range
                       else
                         match ar, br with
                         | Some (al, ah), Some (bl, bh) ->
                             let lo, hi =
                               match op with
                               | "+" -> Int64.add al bl, Int64.add ah bh
                               | "-" -> Int64.sub al bh, Int64.sub ah bl
                               | _ ->
                                   let ps =
                                     List.map
                                       (fun (x, y) -> Int64.mul x y)
                                       [al, bl; al, bh; ah, bl; ah, bh]
                                   in
                                   List.fold_left min Int64.max_int ps,
                                   List.fold_left max Int64.min_int ps
                             in
                             checked exp.exp_loc (lo, hi)
                         | _ ->
                             fail exp.exp_loc
                               "cannot prove native int arithmetic fits i32; use Int32 arithmetic"
                     in
                     const
                       (fun d ->
                         match op with
                         | "+" | "i32_add" -> K.Add_i32 (d, av, bv)
                         | "-" | "i32_sub" -> K.Sub_i32 (d, av, bv)
                         | "i32_div" -> K.Div_i32 (d, av, bv)
                         | "i32_rem" -> K.Rem_i32 (d, av, bv)
                         | _ -> K.Mul_i32 (d, av, bv))
                       range
                 | _ ->
                     let k = kind a.exp_env a.exp_loc a.exp_type in
                     if k = Buffer || k = Unit || k = Shared || k = Mbarrier
                        || k = Tensor_map then
                       fail exp.exp_loc "only scalar comparisons are supported";
                     let c =
                       match op with
                       | "=" -> G.Eq
                       | "<>" -> G.Ne
                       | "<" -> G.Lt
                       | "<=" -> G.Le
                       | ">" -> G.Gt
                       | _ -> G.Ge
                     in
                     if k = Boolean && c <> G.Eq && c <> G.Ne then
                       fail exp.exp_loc "only boolean equality comparisons are supported";
                     let folded_bool =
                      match singleton_int ar, singleton_int br with
                      | Some a, Some b ->
                          Some
                            (match c with
                             | G.Eq -> a = b
                             | G.Ne -> a <> b
                             | G.Lt -> a < b
                             | G.Le -> a <= b
                             | G.Gt -> a > b
                             | G.Ge -> a >= b)
                      | _ -> None
                    in
                    match folded_bool with
                    | Some value -> bool_const s exp value
                    | None -> result (fun d -> K.Compare (d, c, av, bv))))
            | _ -> fail exp.exp_loc ("unsupported GPU call/arity: " ^ op))))
  | Texp_for
      { for_id; for_from; for_to; for_dir = Upto; for_body; _ } ->
      (* OCaml inclusive [for i = lo to hi]. Ir.For is exclusive with step.
         Constant bounds up to [unroll_bounds] iterations are copied, so an
         index such as [4 * i] stays a compile-time register select. Constants
         that folding leaves with no users are dropped at the end of lowering. *)
      let lo, lo_r = scalar for_from.exp_loc (expression s env for_from) in
      let hi, hi_r = scalar for_to.exp_loc (expression s env for_to) in
      let singleton = function
        | Some (a, b) when a = b -> Some a
        | _ -> None
      in
      let emit_dynamic () =
        let limit =
          match hi_r with
          | Some (lo_b, hi_b) when lo_b = hi_b && hi_b < 2147483647L ->
              let d = fresh s in
              emit s exp (K.Const_i32 (d, Int64.to_int hi_b + 1));
              K.Value d
          | Some (lo_b, hi_b) when lo_b = hi_b ->
              fail exp.exp_loc
                "for upper bound is Int32.max_int; an exclusive limit would overflow"
          | _ ->
              let one = fresh s in
              emit s exp (K.Const_i32 (one, 1));
              let d = fresh s in
              emit s exp (K.Add_i32 (d, hi, K.Value one));
              K.Value d
        in
        let ind = fresh s in
        let outer = s.body in
        s.body <- [];
        let env = (for_id, Scalar (K.Value ind, full_range)) :: env in
        unit for_body.exp_loc (expression s env for_body);
        let body = { K.body = List.rev s.body; yield = None } in
        s.body <- outer;
        emit s exp (K.For (ind, lo, limit, 1, body));
        Void
      in
      (match singleton lo_r, singleton hi_r with
       | Some lo_i, Some hi_i ->
           let trips =
             if hi_i < lo_i then 0L else Int64.succ (Int64.sub hi_i lo_i)
           in
           if trips > Int64.of_int unroll_bounds then emit_dynamic ()
           else begin
             let rec step i =
               if i > hi_i then ()
               else begin
                 let d = fresh s in
                 emit s exp (K.Const_i32 (d, Int64.to_int i));
                 let env_i =
                   (for_id, Scalar (K.Value d, Some (i, i))) :: env
                 in
                 unit for_body.exp_loc (expression s env_i for_body);
                 step (Int64.succ i)
               end
             in
             step lo_i;
             Void
           end
       | _ -> emit_dynamic ())
  | Texp_for _ -> fail exp.exp_loc "only ascending for-loops are supported"
  | _ -> fail exp.exp_loc "unsupported GPU expression"

and local_apply s env exp callee xs =
  match callee.exp_desc with
  | Texp_ident { path = Path.Pident id; _ } ->
      (match List.find_opt (fun (key, _) -> Ident.same key id) env with
       | Some (_, Func fn) ->
           let args =
             List.map
               (function
                 | (Optional _ | Position _), _ ->
                     fail exp.exp_loc "optional arguments unsupported"
                 | _, Omitted _ ->
                     fail exp.exp_loc "functions must be fully applied"
                 | label, Arg (arg, _) -> label, expression s env arg)
               xs
           in
           if List.length args <> List.length fn.params then
             fail exp.exp_loc "functions must be fully applied";
           (* Positional arguments share the label [Nolabel], so match those
              in order. Labelled arguments match by name. *)
           let positionals =
             ref (List.filter (fun (label, _) -> label = Nolabel) args)
           in
           let body_env =
             List.fold_left
               (fun env (label, pid) ->
                 let value =
                   match label with
                   | Nolabel ->
                       (match !positionals with
                        | (_, value) :: rest ->
                            positionals := rest;
                            value
                        | [] -> fail exp.exp_loc "missing function argument")
                   | Labelled name ->
                       (match
                          List.find_opt
                            (fun (got, _) -> got = Labelled name) args
                        with
                        | Some (_, value) -> value
                        | None -> fail exp.exp_loc "missing function argument")
                   | Optional _ | Position _ ->
                       fail exp.exp_loc "optional arguments unsupported"
                 in
                 (pid, value) :: env)
               fn.env fn.params
           in
           if !positionals <> [] then
             fail exp.exp_loc "functions must be fully applied";
           Some (expression s body_env fn.body)
       | _ -> None)
  | _ -> None

and conditional s exp condition yes no =
  let outer = s.body in
  let region f =
    s.body <- [];
    let value = f () in
    let yield =
      match value with
      | Void -> None
      | Scalar (a, _) | Const_bool (a, _) -> Some a
      | Func _ -> fail exp.exp_loc "functions cannot be branch results"
      | Memory _ | Shared _ | Mbarrier _ | Tensor_map _ | Wgmma_acc _ ->
          fail exp.exp_loc "non-scalar branch results are unsupported"
    in
    { K.body = List.rev s.body; yield }, value
  in
  let a, av = region yes in
  let b, bv = region no in
  s.body <- outer;
  let dst, value =
    let arm_range = function
      | Scalar (_, r) -> r
      | Const_bool _ -> None
      | _ -> assert false
    in
    match av, bv with
    | Void, Void -> None, Void
    | (Scalar _ | Const_bool _), (Scalar _ | Const_bool _) ->
        let d = fresh s in
        let ar = arm_range av and br = arm_range bv in
        let range =
          match ar, br with
          | Some (al, ah), Some (bl, bh) -> Some (min al bl, max ah bh)
          | _ -> None
        in
        ( Some (d, gpu_ty (kind exp.exp_env exp.exp_loc exp.exp_type))
        , Scalar (K.Value d, range) )
    | _ -> fail exp.exp_loc "conditional arms have incompatible results"
  in
  emit s exp (K.If (dst, condition, a, b));
  value

(* Folding a constant index still emits the operand literals. Drop constants
   that no instruction reads, so an unrolled [4 * g] used only as a register
   select does not leave a trail of dead movs. *)
let elim_unused_consts instrs =
  let used = Hashtbl.create 32 in
  let note_atom a =
    match a with
    | K.Arg _ -> ()
    | K.Value id -> Hashtbl.replace used id ()
  in
  let note_mem = function
    | K.Buffer_arg _ -> ()
    | K.Pointer a -> note_atom a
  in
  let note_opt = function None -> () | Some a -> note_atom a in
  let atoms xs = List.iter note_atom xs in
  let rec note_op op =
    match op with
    | K.Const_bool _ | K.Const_i32 _ | K.Const_f32 _
    | K.Thread_idx_x _ | K.Global_idx_x _ | K.Block_idx_x _ | K.Block_idx_y _
    | K.Shared_alloc _ | K.Barrier_cta | K.Cp_async_commit | K.Cp_async_wait_all
    | K.Mbarrier_alloc _ | K.Mbarrier_set_alloc _
    | K.Fence_proxy_async | K.Wgmma_fence
    | K.Wgmma_commit_group | K.Wgmma_wait_group _ -> ()
    | K.Add_i32 (_, a, b) | K.Sub_i32 (_, a, b) | K.Mul_i32 (_, a, b)
    | K.Min_i32 (_, a, b)
    | K.Div_i32 (_, a, b) | K.Rem_i32 (_, a, b) | K.Compare (_, _, a, b)
    | K.Add_f32 (_, a, b) | K.Mul_f32 (_, a, b) | K.Shared_load_f32 (_, a, b)
    | K.Store_f32 (a, b) -> atoms [a; b]
    | K.Mad_f32 (_, a, b, c) | K.Load_f32_masked (_, a, b, c) ->
        atoms [a; b; c]
    | K.Store_f32_masked (a, b, c, d) -> atoms [a; b; c; d]
    | K.Load_f32x4 (_, _, _, _, a) | K.Load_f32 (_, a) | K.Warp_reduce_sum_f32 (_, a) ->
        note_atom a
    | K.Store_f32x4 (a, b, c, d, e) -> atoms [a; b; c; d; e]
    | K.Shared_load_f32x4 (_, _, _, _, a, b) -> atoms [a; b]
    | K.Shared_store_f32 (a, b, c) -> atoms [a; b; c]
    | K.Shared_store_f32x4 (a, b, c, d, e, f) -> atoms [a; b; c; d; e; f]
    | K.Gep_f32 (_, m, a) -> note_mem m; note_atom a
    | K.Store_grid_leader_f32 (m, a) -> note_mem m; note_atom a
    | K.Cp_async_f32x4 (a, b, m, c) -> atoms [a; b; c]; note_mem m
    | K.If (_, c, a, b) -> note_atom c; note_region a; note_region b
    | K.Mbarrier_init (a, b, p) | K.Mbarrier_arrive_expect_tx (a, b, p) ->
        atoms [a; b]; note_opt p
    | K.Mbarrier_arrive (a, p) -> note_atom a; note_opt p
    | K.Mbarrier_try_wait_parity (a, b) -> atoms [a; b]
    | K.Mbarrier_slot (_, a, b) -> atoms [a; b]
    | K.Tma_load_2d (a, b, c, d, e, f, p) ->
        atoms [a; b; c; d; e; f]; note_opt p
    | K.Gmma_descriptor (_, a, b) -> atoms [a; b]
    | K.Wgmma_mma_tf32 (ids, a, b, _, _) ->
        List.iter (fun id -> Hashtbl.replace used id ()) ids;
        atoms [a; b]
    | K.For (_, lo, limit, _, body) ->
        atoms [lo; limit]; note_region body
  and note_region r =
    note_opt r.K.yield;
    List.iter (fun i -> note_op i.K.op) r.K.body
  in
  List.iter (fun i -> note_op i.K.op) instrs;
  let rec strip instrs =
    List.filter_map (fun i ->
      match i.K.op with
      | K.Const_bool (id, _) | K.Const_i32 (id, _) | K.Const_f32 (id, _)
        when not (Hashtbl.mem used id) -> None
      | K.If (dst, c, a, b) ->
          Some { i with K.op = K.If (dst, c, strip_region a, strip_region b) }
      | K.For (ind, lo, limit, step, body) ->
          Some { i with K.op = K.For (ind, lo, limit, step, strip_region body) }
      | _ -> Some i)
    instrs
  and strip_region r = { r with K.body = strip r.K.body } in
  strip instrs

let threads_attribute binding =
  let open Parsetree in
  let payload_int loc = function
    | PStr
        [{ pstr_desc =
             Pstr_eval
               ( { pexp_desc =
                     Pexp_constant
                       { pconst_desc = Pconst_integer (text, None); _ }
                 ; _ }
               , _ )
         ; _ }] ->
        (match int_of_string_opt text with
         | Some n -> n
         | None -> fail loc "@@gpu.threads integer is out of range")
    | _ ->
        fail loc "@@gpu.threads expects an integer, e.g. [@@gpu.threads 288]"
  in
  match
    List.filter_map
      (fun attr ->
        if attr.attr_name.txt = "gpu.threads" then
          Some (payload_int attr.attr_loc attr.attr_payload)
        else None)
      binding.vb_attributes
  with
  | [] -> None
  | [n] when n >= 1 && n <= 1024 -> Some n
  | [_] -> fail binding.vb_loc "@@gpu.threads must be in 1..1024"
  | _ -> fail binding.vb_loc "duplicate @@gpu.threads"

let entry_formal loc i id (arg : K.argument) =
  let value =
    match arg.slot.ty with
    | G.MemRef (_, _, G.Global) | G.Ptr (_, G.Global) -> Memory (K.Buffer_arg i)
    | G.Tensor_map -> Tensor_map (K.Arg i)
    | G.I32 -> Scalar (K.Arg i, full_range)
    | G.F32 | G.Bool | G.U64 -> Scalar (K.Arg i, None)
    | _ -> fail loc "unsupported kernel argument"
  in
  id, value

(* Structure-level functions defined before [target]. A specialized kernel is
   an application of one of these ([hopper_gemm ~bm:128 ...]); evaluating the
   application inlines the constants, and the returned function is the entry. *)
let prior_functions target structure =
  let env = ref [] in
  let stop = ref false in
  List.iter
    (fun item ->
      if !stop then ()
      else
        match item.str_desc with
        | Tstr_value (_, bindings) ->
            List.iter
              (fun binding ->
                if !stop then ()
                else
                  match binding.vb_pat.pat_desc with
                  | Tpat_var { name; _ } when name.txt = target -> stop := true
                  | Tpat_var { id; _ } ->
                      (match binding.vb_expr.exp_desc with
                       | Texp_function _ ->
                           let scratch = { next = 0; body = [] } in
                           (match expression scratch !env binding.vb_expr with
                            | Func _ as fn -> env := (id, fn) :: !env
                            | _ -> ())
                       | _ -> ())
                  | _ -> ())
              bindings
        | _ -> ())
    structure.str_items;
  !env

let compile target structure =
  let binding =
    List.find_map
      (fun i ->
        match i.str_desc with
        | Tstr_value (_, bs) ->
            List.find_opt
              (fun b ->
                match b.vb_pat.pat_desc with
                | Tpat_var { name; _ } -> name.txt = target
                | _ -> false)
              bs
        | _ -> None)
      structure.str_items
  in
  let binding =
    match binding with
    | Some b -> b
    | None -> fail Location.none "kernel binding not found"
  in
  let tenv = binding.vb_expr.exp_env and loc = binding.vb_expr.exp_loc in
  let rec signature i typ last =
    match Types.get_desc typ with
    | Types.Tpoly (t, _) -> signature i t last
    | Types.Tarrow ((_, am, rm), a, b, _) ->
        let a = { K.index = i; slot = slot tenv loc a am } in
        let rest, result = signature (i + 1) b (Some rm) in
        a :: rest, result
    | _ ->
        match last with
        | Some m -> [], slot tenv loc typ m
        | None -> fail loc "kernel must be a function"
  in
  let args, result = signature 0 binding.vb_pat.pat_type None in
  if result.ty <> G.Unit then fail loc "GPU entry kernels must return unit";
  let finish s =
    { K.name = target
    ; args
    ; result
    ; body = elim_unused_consts (List.rev s.body)
    ; threads = threads_attribute binding
    }
  in
  match binding.vb_expr.exp_desc with
  | Texp_function { params; body = Tfunction_body body; _ } ->
      (* Direct kernels do not see earlier structure bindings. A user
         operator with the same name as a primitive must stay rejected. *)
      let env =
        List.mapi
          (fun i param ->
            let k =
              match param.fp_kind with
              | Tparam_pat p -> kind p.pat_env p.pat_loc p.pat_type
              | _ -> fail loc "optional parameters unsupported"
            in
            let value =
              match k with
              | Buffer -> Memory (K.Buffer_arg i)
              | Tensor_map -> Tensor_map (K.Arg i)
              | Unit -> fail loc "unit kernel argument unsupported"
              | Shared | Mbarrier ->
                  fail loc "shared/mbarrier cannot be kernel formals"
              | Native_int | Int32 -> Scalar (K.Arg i, full_range)
              | Int64 | Float | Boolean -> Scalar (K.Arg i, None)
              | Wgmma_acc -> fail loc "wgmma_acc cannot be a kernel formal"
            in
            param.fp_param, value)
          params
      in
      let s = { next = 0; body = [] } in
      unit body.exp_loc (expression s env body);
      finish s
  | _ ->
      let s = { next = 0; body = [] } in
      let fn =
        match expression s (prior_functions target structure) binding.vb_expr with
        | Func fn -> fn
        | _ ->
            fail loc
              "kernel binding must be a function or a fully applied specialization of one"
      in
      if List.length fn.params <> List.length args then
        fail loc "kernel function arity does not match its type";
      let formals =
        List.mapi
          (fun i ((_, id), arg) -> entry_formal loc i id arg)
          (List.combine fn.params args)
      in
      unit fn.body.exp_loc (expression s (formals @ fn.env) fn.body);
      finish s

let () =
  try
    if Array.length Sys.argv <> 3 then
      fail Location.none "usage: export_typedtree_modes FILE.cmt NAME";
    let _, cmt = Cmt_format.read Sys.argv.(1) in
    match cmt with
    | Some cmt ->
        Load_path.init
          ~auto_include:Load_path.no_auto_include
          ~visible:cmt.cmt_loadpath.visible
          ~hidden:cmt.cmt_loadpath.hidden;
        (match cmt.cmt_annots with
         | Implementation s ->
             print_string (Gpu_metadata.encode (compile Sys.argv.(2) s))
         | _ -> fail Location.none "expected implementation")
    | None -> fail Location.none "missing typedtree"
  with Source_span.Error (loc, msg) ->
    prerr_endline (Source_span.to_string loc ^ ": " ^ msg);
    exit 2
