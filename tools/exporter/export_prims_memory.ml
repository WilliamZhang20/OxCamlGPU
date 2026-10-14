(* Global and CTA-shared memory: scalar and vector loads and stores, shared
   allocation, asynchronous copies, and the CTA barrier. *)
open Typedtree
module K = Kernel_ast
open Typedtree_support
open Export_state

(* Address of element [index] of a global buffer, shared by every access. *)
let element_pointer ctx buffer index =
  let base = arg ctx memory buffer in
  let ix = num ctx index in
  value ctx (fun d -> K.Gep_f32 (d, base, ix))

let load : handler =
  on_positional (fun ctx -> function
    | buffer :: index :: tail ->
        let base = arg ctx memory buffer in
        let ix = num ctx index in
        let bound =
          match ctx.op, tail with
          | "load", [] -> None
          | "load_masked", [n] -> Some (num ctx n)
          | _ -> fail_here ctx "invalid load arity"
        in
        let p = value ctx (fun d -> K.Gep_f32 (d, base, ix)) in
        produce ctx (fun d ->
          match bound with
          | None -> K.Load_f32 (d, p)
          | Some n -> K.Load_f32_masked (d, p, ix, n))
    | _ -> arity ctx)

let store : handler =
  on_positional (fun ctx -> function
    | buffer :: index :: tail ->
        let base = arg ctx memory buffer in
        let ix = num ctx index in
        let bound, v =
          match ctx.op, tail with
          | "store", [v] -> None, v
          | "store_masked", [n; v] -> Some (num ctx n), v
          | _ -> fail_here ctx "invalid store arity"
        in
        let p = value ctx (fun d -> K.Gep_f32 (d, base, ix)) in
        let v = num ctx v in
        emit ctx
          (match bound with
           | None -> K.Store_f32 (p, v)
           | Some n -> K.Store_f32_masked (p, v, ix, n));
        Void
    | _ -> arity ctx)

let store_f32x4 : handler =
  on_positional (fun ctx -> function
    | [buffer; index; a; b; c; d] ->
        let p = element_pointer ctx buffer index in
        let a = num ctx a in
        let b = num ctx b in
        let c = num ctx c in
        let d = num ctx d in
        emit ctx (K.Store_f32x4 (p, a, b, c, d));
        Void
    | _ -> arity ctx)

let shared_load : handler =
  on_positional (fun ctx -> function
    | [smem; index] ->
        let m = arg ctx shared smem in
        let ix = num ctx index in
        produce ctx (fun d -> K.Shared_load_f32 (d, m, ix))
    | _ -> arity ctx)

let shared_store : handler =
  on_positional (fun ctx -> function
    | [smem; index; v] ->
        let m = arg ctx shared smem in
        let ix = num ctx index in
        let v = num ctx v in
        emit ctx (K.Shared_store_f32 (m, v, ix));
        Void
    | _ -> arity ctx)

let shared_store_f32x4 : handler =
  on_positional (fun ctx -> function
    | [smem; index; a; b; c; d] ->
        let m = arg ctx shared smem in
        let ix = num ctx index in
        let a = num ctx a in
        let b = num ctx b in
        let c = num ctx c in
        let d = num ctx d in
        emit ctx (K.Shared_store_f32x4 (m, a, b, c, d, ix));
        Void
    | _ -> arity ctx)

let cp_async_f32x4 : handler =
  on_positional (fun ctx -> function
    | [smem; sidx; buffer; gidx] ->
        let m = arg ctx shared smem in
        let si = num ctx sidx in
        let g = arg ctx memory buffer in
        let gi = num ctx gidx in
        emit ctx (K.Cp_async_f32x4 (m, si, g, gi));
        Void
    | _ -> arity ctx)

let store_grid_leader : handler =
  on_positional (fun ctx -> function
    | [buffer; v] ->
        let p = arg ctx memory buffer in
        let v = num ctx v in
        emit ctx (K.Store_grid_leader_f32 (p, v));
        Void
    | _ -> arity ctx)

let cp_async_wait : handler = unit_effect K.Cp_async_wait_all

(* [~rows] and [~cols] must be compile-time constants. The tile's element type
   is read off the allocation's own type, which inference has already pinned
   from how the tile is used, normally the typed tensor_map that fills it. *)
let shared_alloc : handler =
 fun ctx xs ->
  let layout = if ctx.op = "shared_wgmma" then `Wgmma else `Plain in
  let element =
    match kind ctx.exp.exp_env ctx.exp.exp_loc ctx.exp.exp_type with
    | Typedtree_support.Shared element -> element
    | _ -> fail_here ctx "shared allocation must have a tile type"
  in
  match labelled ctx.exp.exp_loc ~required:["rows"; "cols"] xs with
  | [rows; cols], terminator ->
      Option.iter (unit_arg ctx) terminator;
      let rows = arg ctx const_extent rows in
      let cols = arg ctx const_extent cols in
      Shared
        (value ctx (fun d -> K.Shared_alloc (d, rows, cols, element, layout)))
  | _ -> fail_here ctx "shared expects ~rows and ~cols"

(* [let (a, b, c, d) = load_f32x4 buf i] and its shared-memory twin define four
   registers at once, so they bind names rather than return a value. *)
let load_vector4 ctx xs =
  match positional ctx.exp.exp_loc xs with
  | [buffer; index] when ctx.op = "load_f32x4" ->
      let p = element_pointer ctx buffer index in
      let ia, ib, ic, id = fresh_four ctx in
      emit ctx (K.Load_f32x4 (ia, ib, ic, id, p));
      ia, ib, ic, id
  | [smem; index] when ctx.op = "shared_load_f32x4" ->
      let m = arg ctx shared smem in
      let ix = num ctx index in
      let ia, ib, ic, id = fresh_four ctx in
      emit ctx (K.Shared_load_f32x4 (ia, ib, ic, id, m, ix));
      ia, ib, ic, id
  | _ -> fail_here ctx "invalid f32x4 arity"

(* The vector loads only make sense in a four-name tuple binding. *)
let vector_load_needs_binding : handler =
 fun ctx _ -> fail_here ctx (ctx.op ^ " must bind four names")

let prims : (string * handler) list =
  [ "load", load; "load_masked", load
  ; "store", store; "store_masked", store
  ; "store_f32x4", store_f32x4
  ; "load_f32x4", vector_load_needs_binding
  ; "shared_load_f32x4", vector_load_needs_binding
  ; "shared", shared_alloc; "shared_wgmma", shared_alloc
  ; "shared_load", shared_load
  ; "shared_store", shared_store
  ; "shared_store_f32x4", shared_store_f32x4
  ; "barrier_cta", unit_effect K.Barrier_cta
  ; "cp_async_f32x4", cp_async_f32x4
  ; "cp_async_commit", unit_effect K.Cp_async_commit
  ; "cp_async_wait", cp_async_wait
  ; "store_grid_leader", store_grid_leader
  ]
