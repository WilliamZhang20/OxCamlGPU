(* Hopper-generation primitives: mbarriers, TMA loads, async-proxy fences, and
   the warpgroup matrix-multiply accumulator. *)
open Typedtree
module K = Kernel_ast
open Typedtree_support
open Export_state

(* Optional election predicate: the [_elect] variants take it as a last
   argument and the plain variants do not, so one handler serves both. *)
let elect_variant ctx = String.ends_with ~suffix:"_elect" ctx.op

(* ---- mbarrier ---- *)

let mbarrier_alloc : handler =
  on_positional (fun ctx -> function
    | [u] ->
        unit_arg ctx u;
        Mbarrier (value ctx (fun d -> K.Mbarrier_alloc d))
    | _ -> arity ctx)

let mbarrier_set : handler =
  on_positional (fun ctx -> function
    | [count] ->
        let count = arg ctx const_extent count in
        Mbarrier (value ctx (fun d -> K.Mbarrier_set_alloc (d, count)))
    | _ -> arity ctx)

let mbarrier_slot : handler =
  on_positional (fun ctx -> function
    | [set; index] ->
        let set = arg ctx mbarrier set in
        let ix = num ctx index in
        Mbarrier (value ctx (fun d -> K.Mbarrier_slot (d, set, ix)))
    | _ -> arity ctx)

let mbarrier_init : handler =
  on_positional (fun ctx -> function
    | [m; c] when not (elect_variant ctx) ->
        let m = arg ctx mbarrier m in
        let c = num ctx c in
        emit ctx (K.Mbarrier_init (m, c, None));
        Void
    | [m; c; p] when elect_variant ctx ->
        let m = arg ctx mbarrier m in
        let c = num ctx c in
        let p = num ctx p in
        emit ctx (K.Mbarrier_init (m, c, Some p));
        Void
    | _ -> arity ctx)

let mbarrier_arrive_expect_tx : handler =
  on_positional (fun ctx -> function
    | [m; tx] when not (elect_variant ctx) ->
        let m = arg ctx mbarrier m in
        let tx = num ctx tx in
        emit ctx (K.Mbarrier_arrive_expect_tx (m, tx, None));
        Void
    | [m; tx; p] when elect_variant ctx ->
        let m = arg ctx mbarrier m in
        let tx = num ctx tx in
        let p = num ctx p in
        emit ctx (K.Mbarrier_arrive_expect_tx (m, tx, Some p));
        Void
    | _ -> arity ctx)

let mbarrier_arrive : handler =
  on_positional (fun ctx -> function
    | [m] when not (elect_variant ctx) ->
        let m = arg ctx mbarrier m in
        emit ctx (K.Mbarrier_arrive (m, None));
        Void
    | [m; p] when elect_variant ctx ->
        let m = arg ctx mbarrier m in
        let p = num ctx p in
        emit ctx (K.Mbarrier_arrive (m, Some p));
        Void
    | _ -> arity ctx)

let mbarrier_try_wait_parity : handler =
  on_positional (fun ctx -> function
    | [m; parity] ->
        let m = arg ctx mbarrier m in
        let parity = num ctx parity in
        emit ctx (K.Mbarrier_try_wait_parity (m, parity));
        Void
    | _ -> arity ctx)

(* ---- TMA ---- *)

let tma_load_2d : handler =
  on_positional (fun ctx -> function
    | sm :: idx :: tmap :: c0 :: c1 :: mbar :: tail
      when List.length tail = (if elect_variant ctx then 1 else 0) ->
        let sm = arg ctx shared sm in
        let idx = num ctx idx in
        let tmap = arg ctx tensor_map tmap in
        let c0 = num ctx c0 in
        let c1 = num ctx c1 in
        let mbar = arg ctx mbarrier mbar in
        let predicate = match tail with [p] -> Some (num ctx p) | _ -> None in
        emit ctx (K.Tma_load_2d (sm, idx, tmap, c0, c1, mbar, predicate));
        Void
    | _ -> arity ctx)

(* ---- warpgroup MMA ---- *)

let wgmma_wait_group : handler =
  on_positional (fun ctx -> function
    | [n] ->
        let n = arg ctx const_nat n in
        emit ctx (K.Wgmma_wait_group n);
        Void
    | _ -> arity ctx)

(* An accumulator for an m64nNk8 tile holds N/2 registers per thread. *)
let wgmma_acc_alloc : handler =
 fun ctx xs ->
  match labelled ctx.exp.exp_loc ~required:["n"] xs with
  | [n], terminator ->
      Option.iter (unit_arg ctx) terminator;
      let n = arg ctx const_extent n in
      if n < 8 || n > 256 || n mod 8 <> 0 then
        fail_here ctx "wgmma_acc ~n must be in {8,16,...,256}";
      Wgmma_acc
        (List.init (n / 2) (fun _ ->
           let d = fresh ctx.s in
           emit ctx (K.Const_f32 (d, 0.));
           d))
  | _ -> fail_here ctx "wgmma_acc expects ~n"

let gmma_descriptor : handler =
  on_positional (fun ctx -> function
    | [smem; index] ->
        let m = arg ctx shared smem in
        let ix = num ctx index in
        produce ctx (fun d -> K.Gmma_descriptor (d, m, ix))
    | _ ->
        fail_here ctx
          "gmma_descriptor expects a shared tile and an element offset")

let wgmma_mma : handler =
 fun ctx xs ->
  let labelled_args, _ = labelled ctx.exp.exp_loc ~required:["n"; "scale"] xs in
  let positionals =
    List.filter_map
      (function Types.Nolabel, Arg (e, _) -> Some e | _ -> None)
      xs
  in
  match positionals, labelled_args with
  | [acc; da; db], [n; scale] ->
      (* One name, and the operands' element type picks the instruction. The
         OCaml type already forced A and B to agree, so reading either
         descriptor is enough. *)
      let element =
        match kind da.exp_env da.exp_loc da.exp_type with
        | Gmma_desc element -> element
        | _ ->
            fail da.exp_loc
              "wgmma_mma operands must come from Gpu.gmma_descriptor"
      in
      let ids = arg ctx wgmma_acc acc in
      let da = num ctx da in
      let db = num ctx db in
      let n = arg ctx const_extent n in
      let scale =
        match scale.exp_desc with
        | Texp_construct (_, { cstr_name = "true"; _ }, _, [], _) -> true
        | Texp_construct (_, { cstr_name = "false"; _ }, _, [], _) -> false
        | _ -> fail scale.exp_loc "scale must be true or false"
      in
      emit ctx (K.Wgmma_mma (element, ids, da, db, n, scale));
      Void
  | _ ->
      fail_here ctx "wgmma_mma expects acc, desc_a, desc_b, ~n, ~scale"

(* A compile-time-constant register index into an accumulator. *)
let constant_register ctx ids idx =
  match eval ctx idx with
  | Scalar (_, Some (lo, hi))
    when lo = hi && lo >= 0L && Int64.to_int lo < List.length ids ->
      Int64.to_int lo
  | _ -> fail idx.exp_loc (ctx.op ^ " index must be a constant in range")

let wgmma_acc_get : handler =
  on_positional (fun ctx -> function
    | [acc; idx] ->
        let ids = arg ctx wgmma_acc acc in
        let i = constant_register ctx ids idx in
        Scalar (K.Value (List.nth ids i), None)
    | _ -> arity ctx)

(* Where accumulator register [i] lives inside this warpgroup's m64 x N tile.
   wgmma.m64nNk8 puts lane [l] of warp [w] at
     row = w*16 + l/4 + 8*((i mod 4)/2)
     col = (l mod 4)*2 + (i mod 2) + 8*(i/4)
   which is hardware layout, not a schedule choice, so the compiler emits it
   rather than asking an author to rewrite it per kernel. Which 64 rows a
   warpgroup owns stays with the author. *)
let wgmma_acc_position : handler =
  on_positional (fun ctx -> function
    | [acc; idx] ->
        let ids = arg ctx wgmma_acc acc in
        let i = constant_register ctx ids idx in
        let { add; mul; div; rem; konst; _ } = index_arith ctx in
        let tid = thread_idx ctx in
        let lane_wg = rem tid (konst 128) in
        let lane = rem lane_wg (konst 32) in
        let base, offset =
          if ctx.op = "wgmma_acc_row" then
            ( add (mul (div lane_wg (konst 32)) (konst 16)) (div lane (konst 4))
            , 8 * (i mod 4 / 2) )
          else
            ( mul (rem lane (konst 4)) (konst 2)
            , (i mod 2) + 8 * (i / 4) )
        in
        let position = if offset = 0 then base else add base (konst offset) in
        Scalar (position, full_range)
    | _ -> arity ctx)

let prims : (string * handler) list =
  [ "mbarrier", mbarrier_alloc
  ; "mbarrier_set", mbarrier_set
  ; "mbarrier_slot", mbarrier_slot
  ; "mbarrier_init", mbarrier_init
  ; "mbarrier_init_elect", mbarrier_init
  ; "mbarrier_arrive_expect_tx", mbarrier_arrive_expect_tx
  ; "mbarrier_arrive_expect_tx_elect", mbarrier_arrive_expect_tx
  ; "mbarrier_arrive", mbarrier_arrive
  ; "mbarrier_arrive_elect", mbarrier_arrive
  ; "mbarrier_try_wait_parity", mbarrier_try_wait_parity
  ; "tma_load_2d", tma_load_2d
  ; "tma_load_2d_elect", tma_load_2d
  ; "fence_proxy_async", unit_effect K.Fence_proxy_async
  ; "wgmma_fence", unit_effect K.Wgmma_fence
  ; "wgmma_commit_group", unit_effect K.Wgmma_commit_group
  ; "wgmma_wait_group", wgmma_wait_group
  ; "wgmma_acc", wgmma_acc_alloc
  ; "wgmma_acc_get", wgmma_acc_get
  ; "wgmma_acc_row", wgmma_acc_position
  ; "wgmma_acc_col", wgmma_acc_position
  ; "gmma_descriptor", gmma_descriptor
  ; "wgmma_mma", wgmma_mma
  ]
