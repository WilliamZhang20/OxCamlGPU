(* Folding a constant index still emits the operand literals. Drop constants
   that no instruction reads, so an unrolled [4 * g] used only as a register
   select does not leave a trail of dead movs. *)
module K = Kernel_ast

let used_ids instrs =
  let used = Hashtbl.create 32 in
  let note_id id = Hashtbl.replace used id () in
  let note_atom = function K.Arg _ -> () | K.Value id -> note_id id in
  let atoms = List.iter note_atom in
  let note_opt = Option.iter note_atom in
  let note_mem = function
    | K.Buffer_arg _ -> ()
    | K.Pointer a -> note_atom a
  in
  let rec note_op = function
    | K.Const_bool _ | K.Const_i32 _ | K.Const_f32 _
    | K.Thread_idx_x _ | K.Global_idx_x _ | K.Block_idx_x _ | K.Block_idx_y _
    | K.Shared_alloc _ | K.Barrier_cta | K.Cp_async_commit | K.Cp_async_wait_all
    | K.Mbarrier_alloc _ | K.Mbarrier_set_alloc _
    | K.Fence_proxy_async | K.Wgmma_fence
    | K.Wgmma_commit_group | K.Wgmma_wait_group _ -> ()
    | K.Add_i32 (_, a, b) | K.Sub_i32 (_, a, b) | K.Mul_i32 (_, a, b)
    | K.Min_i32 (_, a, b)
    | K.Div_i32 (_, a, b) | K.Rem_i32 (_, a, b) | K.Compare (_, _, a, b)
    | K.Binary_f32 (_, _, a, b)
    | K.Add_f32 (_, a, b) | K.Mul_f32 (_, a, b) | K.Shared_load_f32 (_, a, b)
    | K.Store_f32 (a, b) -> atoms [a; b]
    | K.Unary_f32 (_, _, a) | K.F32_of_i32 (_, a) -> note_atom a
    | K.Mad_f32 (_, a, b, c) | K.Load_f32_masked (_, a, b, c) ->
        atoms [a; b; c]
    | K.Store_f32_masked (a, b, c, d) -> atoms [a; b; c; d]
    | K.Load_f32x4 (_, _, _, _, a) | K.Load_f32 (_, a)
    | K.Warp_reduce_sum_f32 (_, a) -> note_atom a
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
    | K.Wgmma_mma (_, ids, a, b, _, _) -> List.iter note_id ids; atoms [a; b]
    | K.For (_, lo, limit, _, body) -> atoms [lo; limit]; note_region body
  and note_region r =
    note_opt r.K.yield;
    List.iter (fun i -> note_op i.K.op) r.K.body
  in
  List.iter (fun i -> note_op i.K.op) instrs;
  used

let eliminate_unused_constants instrs =
  let used = used_ids instrs in
  let rec strip instrs =
    List.filter_map
      (fun i ->
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
