open Ir
let require condition message = if not condition then invalid_arg message
let is_pointer = Gpu_type.is_global_f32_buffer
(* A warp reduction input is a 1D f32 tensor laid out so that one 32-lane
   subgroup of one CTA owns every element. Both register layouts answer "where
   does logical index i live"; only the accessor differs, so the two arms pass
   their own and share the test. *)
let is_warp_reduction_input value =
  let warp_local extent ~lanes ~subgroups ~per_lane coordinate =
    lanes = [32] && subgroups = [1] && extent = 32 * List.hd per_lane &&
    List.for_all (fun index ->
      let h = coordinate index in
      h.Layout.subgroup_id = 0 && h.Layout.cta = [0])
      (List.init extent Fun.id)
  in
  match value.ty, value.layout with
  | Gpu_type.Tensor ([Gpu_type.Static extent], Gpu_type.Float32),
    Some (Layout.Register r) ->
      warp_local extent ~lanes:r.lanes_per_subgroup
        ~subgroups:r.subgroups_per_cta ~per_lane:r.elements_per_lane
        (fun index -> Layout.logical_to_hardware r [index])
  | Gpu_type.Tensor ([Gpu_type.Static extent], Gpu_type.Float32),
    Some (Layout.Mapped_register r) ->
      warp_local extent ~lanes:r.lanes_per_subgroup
        ~subgroups:r.subgroups_per_cta ~per_lane:r.elements_per_lane
        (fun index -> Layout.mapped_to_hardware r [index])
  | _ -> false

(* Target legalization belongs with the PTX backend: this is a strategy choice
   for this target, not a source/semantic IR pass. *)
let lower_unchecked kernel =
  let next_id = ref (1 + List.fold_left max (-1)
      (List.map (fun arg -> arg.value.id) kernel.args @
       List.concat_map (fun instr -> List.map (fun v -> v.id) (results instr)) (Ir.flatten kernel.body))) in
  let fresh ty = let id = !next_id in incr next_id; make_value id ty in
  let fresh_shared shape =
    let id = !next_id in incr next_id;
    let order = match shape with
      | [_] -> [0]
      | [_; _] -> [1; 0]
      | _ -> invalid_arg "shared remap staging supports rank-1 or rank-2 tiles only" in
    make_value ~permission:Read_write ~locality:Local
      ~layout:(Layout.Shared { vector_width=1; order; swizzle=Layout.No_swizzle })
      id (Gpu_type.MemRef (shape, Gpu_type.Float32, Gpu_type.Shared)) in
  let labels = ref 0 in
  let label () = let n = !labels in incr labels;n in
  let rec lower body = List.concat_map (function
      | If(dst,c,a,b) ->
          let yes=label() and no=label() and done_=label() in
          let edge r = match dst,r.yield with Some d,Some v->[Ptx_ir.Move(d,v)] | None,None->[] | _->invalid_arg "invalid branch yield" in
          [Ptx_ir.Branch(c,yes,no);Ptx_ir.Label yes] @ lower a.body @ edge a @
          [Ptx_ir.Jump done_;Ptx_ir.Label no] @ lower b.body @ edge b @ [Ptx_ir.Label done_]
      | If_uni(dst,c,a,b) ->
          let yes=label() and no=label() and done_=label() in
          let edge r = match dst,r.yield with Some d,Some v->[Ptx_ir.Move(d,v)] | None,None->[] | _->invalid_arg "invalid branch yield" in
          [Ptx_ir.Branch_uni(c,yes,no);Ptx_ir.Label yes] @ lower a.body @ edge a @
          [Ptx_ir.Jump_uni done_;Ptx_ir.Label no] @ lower b.body @ edge b @ [Ptx_ir.Label done_]
      | For (induction, limit, step, loop_body) ->
          let head = label () and body_l = label () and done_l = label () in
          let pred = fresh Gpu_type.Bool in
          let step_v = fresh Gpu_type.I32 in
          [ Ptx_ir.Label head
          ; Ptx_ir.Compare (pred, Gpu_type.Lt, induction, limit)
          ; Ptx_ir.Branch_uni (pred, body_l, done_l)
          ; Ptx_ir.Label body_l ]
          @ lower loop_body
          @ [ Ptx_ir.Const_i32 (step_v, step)
            ; Ptx_ir.Add_i32 (induction, induction, step_v)
            ; Ptx_ir.Jump_uni head
            ; Ptx_ir.Label done_l ]
      | Const_bool(v,b) -> [Ptx_ir.Const_bool(v,b)]
      | Compare(d,c,a,b) -> [Ptx_ir.Compare(d,c,a,b)]
      | Sub_i32(d,a,b) -> [Ptx_ir.Sub_i32(d,a,b)]
      | Mul_i32(d,a,b) -> [Ptx_ir.Mul_i32(d,a,b)]
      | Div_i32(d,a,b) -> [Ptx_ir.Div_i32(d,a,b)]
      | Rem_i32(d,a,b) -> [Ptx_ir.Rem_i32(d,a,b)]
      | Reduce_sum_f32 (dst, src) ->
          if not (is_warp_reduction_input src) then
            invalid_arg "PTX reduction requires a 1D f32 tile distributed across one 32-lane subgroup";
          let lane_value = fresh Gpu_type.F32 in
          [Ptx_ir.Tensor_lane_f32 (lane_value, src);
           Ptx_ir.Warp_butterfly_sum_f32 (dst, lane_value)]
      | Warp_sum_f32 (dst, src) -> [Ptx_ir.Warp_butterfly_sum_f32 (dst, src)]
      | Const_i32 (v,n) -> [Ptx_ir.Const_i32(v,n)]
      | Const_f32 (v,n) -> [Ptx_ir.Const_f32(v,n)]
      | Add_i32 (v,a,b) -> [Ptx_ir.Add_i32(v,a,b)]
      | Add_f32 (v,a,b) -> [Ptx_ir.Add_f32(v,a,b)]
      | Mul_f32 (v,a,b) -> [Ptx_ir.Mul_f32(v,a,b)]
      | Mad_f32 (dst, a, b, c) ->
          [Ptx_ir.Mad_f32 (dst, a, b, c)]
      | Scale_tensor_f32 (v,a,b) -> [Ptx_ir.Tensor_scale_f32(v,a,b)]
      | Mul_tensor_f32 (v,a,b) -> [Ptx_ir.Tensor_mul_f32(v,a,b)]
      | Remap_tensor (dst, src) ->
          (match src.ty with
           | Gpu_type.Tensor (shape, Gpu_type.Float32) ->
               let scratch=fresh_shared shape in
               [Ptx_ir.Shared_alloc scratch; Ptx_ir.Tile_store_f32(scratch,src);
                Ptx_ir.Barrier_cta; Ptx_ir.Tile_load_f32(dst,scratch)]
           | _ -> invalid_arg "remap requires an f32 tensor")
      | Thread_idx_x v -> [Ptx_ir.Thread_idx_x v]
      | Global_idx_x v -> [Ptx_ir.Global_idx_x v]
      | Block_idx_x v -> [Ptx_ir.Block_idx_x v]
      | Block_idx_y v -> [Ptx_ir.Block_idx_y v]
      | Gep_f32 (v,p,i) -> [Ptx_ir.Gep_f32(v,p,i)]
      | Load_f32 (v,p) -> [Ptx_ir.Load_f32(v,p)]
      | Load_f32_masked (v,p,i,n) -> [Ptx_ir.Load_f32_masked(v,p,i,n)]
      | Load_f32x4 (a,b,c,d,p) -> [Ptx_ir.Load_f32x4(a,b,c,d,p)]
      | Store_f32 (p,v) -> [Ptx_ir.Store_f32(p,v)]
      | Store_f32_masked (p,v,i,n) -> [Ptx_ir.Store_f32_masked(p,v,i,n)]
      | Store_f32x2 (p,a,b) -> [Ptx_ir.Store_f32x2(p,a,b)]
      | Store_f32x4 (p,a,b,c,d) -> [Ptx_ir.Store_f32x4(p,a,b,c,d)]
      | Shared_alloc v -> [Ptx_ir.Shared_alloc v]
      | Shared_load_f32 (v,p,i) -> [Ptx_ir.Shared_load_f32(v,p,i)]
      | Shared_load_f32x4 (a,b,c,d,p,i) -> [Ptx_ir.Shared_load_f32x4(a,b,c,d,p,i)]
      | Shared_store_f32 (p,v,i) -> [Ptx_ir.Shared_store_f32(p,v,i)]
      | Shared_store_f32x4 (p,a,b,c,d,i) -> [Ptx_ir.Shared_store_f32x4(p,a,b,c,d,i)]
      | Cp_async_shared_f32x4 (p,i,g) -> [Ptx_ir.Cp_async_shared_f32x4(p,i,g)]
      | Cp_async_commit -> [Ptx_ir.Cp_async_commit]
      | Cp_async_wait_all -> [Ptx_ir.Cp_async_wait_all]
      | Mbarrier_init (m,c,p) -> [Ptx_ir.Mbarrier_init(m,c,p)]
      | Mbarrier_arrive_expect_tx (m,t,p) -> [Ptx_ir.Mbarrier_arrive_expect_tx(m,t,p)]
      | Mbarrier_arrive (m,p) -> [Ptx_ir.Mbarrier_arrive(m,p)]
      | Mbarrier_try_wait_parity (m,p) -> [Ptx_ir.Mbarrier_try_wait_parity(m,p)]
      | Mbarrier_slot (d,set,ix) -> [Ptx_ir.Mbarrier_slot(d,set,ix)]
      | Tma_load_2d (s,i,t,c0,c1,m,p) -> [Ptx_ir.Tma_load_2d(s,i,t,c0,c1,m,p)]
      | Fence_proxy_async -> [Ptx_ir.Fence_proxy_async]
      | Gmma_descriptor (d, s, i, fields) ->
          [Ptx_ir.Gmma_descriptor (d, s, i, fields)]
      | Wgmma_fence -> [Ptx_ir.Wgmma_fence]
      | Wgmma_mma_tf32 (acc,da,db,n,s) ->
          [Ptx_ir.Wgmma_mma_tf32 (acc,da,db,n,s)]
      | Wgmma_commit_group -> [Ptx_ir.Wgmma_commit_group]
      | Wgmma_wait_group n -> [Ptx_ir.Wgmma_wait_group n]
      | Load_tensor (v,p) -> [Ptx_ir.Tile_load_f32(v,p)]
      | Store_tensor (p,v) -> [Ptx_ir.Tile_store_f32(p,v)]
      | Load_tensor_masked (v,p,rows,cols) -> [Ptx_ir.Tile_load_f32_masked(v,p,rows,cols)]
      | Store_tensor_masked (p,v,rows,cols) -> [Ptx_ir.Tile_store_f32_masked(p,v,rows,cols)]
      | Store_f32_grid_leader (p,v) -> [Ptx_ir.Store_grid_leader_f32(p,v)]
      | Barrier Execution.Cta -> [Ptx_ir.Barrier_cta]
      | Barrier _ -> invalid_arg "PTX target lowering supports CTA barriers only"
      | Return None -> [Ptx_ir.Return]
      | Return (Some _) -> invalid_arg "PTX kernel entry points cannot return a value") body in
  let body=lower kernel.body in
  let uses_subgroup = List.exists (function
    | Ptx_ir.Tensor_lane_f32 _ | Ptx_ir.Warp_butterfly_sum_f32 _ -> true
    | Ptx_ir.Tile_load_f32 _ | Ptx_ir.Tile_store_f32 _ | Ptx_ir.Shared_alloc _ -> true
    | Ptx_ir.Tile_load_f32_masked _ | Ptx_ir.Tile_store_f32_masked _ -> true
    | _ -> false) body in
  let shared_elems =
    List.fold_left
      (fun acc -> function
        | Ptx_ir.Shared_alloc memref ->
            (match memref.ty with
             | Gpu_type.MemRef (shape, _, Gpu_type.Shared) ->
                 max acc (Gpu_type.shape_elements shape)
             | _ -> acc)
        | _ -> acc)
      0 body
  in
  let threads_per_cta =
    match kernel.threads_per_cta with
    | Some n when n > 0 -> Some (n, 1, 1)
    | _ ->
      (* Warp-local shared tiles stay at 32 threads; multi-KB CTA tiles use 256. *)
      if shared_elems > 256 then Some (256, 1, 1)
      else if uses_subgroup then Some (32, 1, 1)
      else None
  in
  (* Reject allocations the window planner cannot size, then take the size
     from the planner itself so the contract matches what PTX declares. *)
  List.iter (function
    | Ptx_ir.Shared_alloc memref ->
        (match memref.ty, memref.layout with
         | Gpu_type.MemRef (_, _, Gpu_type.Shared), Some _ -> ()
         | _ -> invalid_arg "shared allocation needs a statically sized shared MemRef layout")
    | _ -> ()) body;
  let shared_bytes = (Ptx_ir.shared_plan kernel.name body).Ptx_ir.total_bytes in
  { Ptx_ir.name=kernel.name; args=kernel.args; body;
    launch={threads_per_cta; shared_bytes} }

let require_register_layout value = match value.ty, value.layout with
  | Gpu_type.Tensor (shape, Gpu_type.Float32), Some (Layout.Register r as layout)
    when List.fold_left ( * ) 1 r.lanes_per_subgroup=32 &&
         List.fold_left ( * ) 1 r.subgroups_per_cta=1 &&
         (match Layout.validate shape layout with Ok () -> true | Error _ -> false) -> ()
  | Gpu_type.Tensor (shape, Gpu_type.Float32), Some (Layout.Mapped_register r as layout)
    when List.fold_left ( * ) 1 r.lanes_per_subgroup=32 &&
         List.fold_left ( * ) 1 r.subgroups_per_cta=1 &&
         (match Layout.validate shape layout with Ok () -> true | Error _ -> false) -> ()
  | _ -> invalid_arg "PTX tile movement supports only f32 tensors distributed over one 32-lane warp"

let require_shared_layout value = match value.ty, value.layout with
  | Gpu_type.MemRef ([Gpu_type.Static _], Gpu_type.Float32, Gpu_type.Shared),
    Some (Layout.Shared { vector_width=1; order=[0]; swizzle=Layout.No_swizzle }) -> ()
  | Gpu_type.MemRef ([Gpu_type.Static _; Gpu_type.Static _], Gpu_type.Float32, Gpu_type.Shared),
    Some (Layout.Shared { vector_width=1; order=[0;1]|[1;0]; swizzle=Layout.No_swizzle }) -> ()
  | _ -> invalid_arg "PTX tile movement supports only unpadded, unswizzled 1D/2D f32 shared layouts"

let require_global_tile value = match value.ty, value.layout with
  | Gpu_type.MemRef ([Gpu_type.Static _], Gpu_type.Float32, Gpu_type.Global), None -> ()
  | Gpu_type.MemRef ([Gpu_type.Static _; Gpu_type.Static _], Gpu_type.Float32, Gpu_type.Global),
    (None | Some (Layout.Global _)) -> ()
  | _ -> invalid_arg "PTX tile movement supports only contiguous or explicitly strided global f32 MemRefs"


let lower kernel =
  Verifier.verify_exn kernel;
  let kernel=lower_unchecked kernel in
  let shared_symbols = Hashtbl.create 4 in
  List.iter (function
    | Ptx_ir.Shared_alloc memref ->
        (match memref.ty, memref.layout with
         | Gpu_type.MemRef (shape, dtype, Gpu_type.Shared),
           Some (Layout.Shared _ | Layout.Padded_shared _ as layout) ->
             (match Layout.validate shape layout with
              | Ok () -> ignore (Gpu_type.storage_bytes shape dtype)
              | Error message -> invalid_arg ("invalid shared allocation layout: " ^ message))
         | _ -> invalid_arg "shared allocation needs a statically shaped shared MemRef layout");
        Hashtbl.replace shared_symbols memref.id ("__shared_" ^ string_of_int memref.id)
    | Ptx_ir.Tile_load_f32 (tensor, memref) ->
        require_register_layout tensor;
        (match memref.ty with
         | Gpu_type.MemRef (_, _, Gpu_type.Shared) -> require_shared_layout memref
         | Gpu_type.MemRef (_, _, Gpu_type.Global) -> require_global_tile memref
         | _ -> invalid_arg "PTX tile load source must be global or shared memory")
    | Ptx_ir.Tile_store_f32 (memref, tensor) ->
        require_register_layout tensor;
        (match memref.ty with
         | Gpu_type.MemRef (_, _, Gpu_type.Shared) -> require_shared_layout memref
         | Gpu_type.MemRef (_, _, Gpu_type.Global) -> require_global_tile memref
        | _ -> invalid_arg "PTX tile store destination must be global or shared memory")
    | Ptx_ir.Tile_load_f32_masked (tensor, memref, rows, cols) ->
        require_register_layout tensor;
        require (rows.ty = Gpu_type.I32 && cols.ty = Gpu_type.I32) "masked tile bounds must be i32";
        (match memref.ty with
         | Gpu_type.MemRef ([_; _], Gpu_type.Float32, Gpu_type.Shared) -> require_shared_layout memref
         | Gpu_type.MemRef ([_; _], Gpu_type.Float32, Gpu_type.Global) -> require_global_tile memref
         | _ -> invalid_arg "masked tile load requires a rank-2 f32 MemRef")
    | Ptx_ir.Tile_store_f32_masked (memref, tensor, rows, cols) ->
        require_register_layout tensor;
        require (rows.ty = Gpu_type.I32 && cols.ty = Gpu_type.I32) "masked tile bounds must be i32";
        (match memref.ty with
         | Gpu_type.MemRef ([_; _], Gpu_type.Float32, Gpu_type.Shared) -> require_shared_layout memref
         | Gpu_type.MemRef ([_; _], Gpu_type.Float32, Gpu_type.Global) -> require_global_tile memref
         | _ -> invalid_arg "masked tile store requires a rank-2 f32 MemRef")
    | Ptx_ir.Mul_f32 (dst, a, c)
    | Ptx_ir.Tensor_scale_f32 (dst, a, c)
    | Ptx_ir.Tensor_mul_f32 (dst, a, c) ->
        (match dst.ty with Gpu_type.Tensor _ ->
           require_register_layout dst; require_register_layout a
         | _ -> ());
        (match c.ty with Gpu_type.Tensor _ -> require_register_layout c | _ -> ())
    | Ptx_ir.Tensor_lane_f32 (_, src) -> require_register_layout src
    | Ptx_ir.Warp_butterfly_sum_f32 (_, src) ->
        require (src.ty = Gpu_type.F32) "warp sum requires f32 lane values"
    | Ptx_ir.Store_grid_leader_f32 (ptr, value) ->
        require (is_pointer ptr.ty) "grid-leader store requires a global f32 buffer";
        require (value.ty = Gpu_type.F32) "grid-leader store value must be f32"
    | Ptx_ir.Const_bool _ | Ptx_ir.Sub_i32 _ | Ptx_ir.Mul_i32 _ | Ptx_ir.Div_i32 _
    | Ptx_ir.Rem_i32 _ | Ptx_ir.Compare _
    | Ptx_ir.Label _ | Ptx_ir.Branch _ | Ptx_ir.Branch_uni _ | Ptx_ir.Jump _
    | Ptx_ir.Jump_uni _ | Ptx_ir.Move _
    | Ptx_ir.Barrier_cta -> ()
    | Ptx_ir.Const_i32 _ | Ptx_ir.Const_f32 _ | Ptx_ir.Add_i32 _ | Ptx_ir.Add_f32 _
    | Ptx_ir.Mad_f32 _
    | Ptx_ir.Thread_idx_x _ | Ptx_ir.Global_idx_x _ | Ptx_ir.Block_idx_x _ | Ptx_ir.Block_idx_y _
    | Ptx_ir.Gep_f32 _
    | Ptx_ir.Load_f32 _ | Ptx_ir.Load_f32_masked _ | Ptx_ir.Load_f32x4 _
    | Ptx_ir.Shared_load_f32 _ | Ptx_ir.Shared_load_f32x4 _
    | Ptx_ir.Shared_store_f32 _ | Ptx_ir.Shared_store_f32x4 _
    | Ptx_ir.Cp_async_shared_f32x4 _ | Ptx_ir.Cp_async_commit | Ptx_ir.Cp_async_wait_all
    | Ptx_ir.Mbarrier_init _ | Ptx_ir.Mbarrier_arrive_expect_tx _
    | Ptx_ir.Mbarrier_arrive _ | Ptx_ir.Mbarrier_try_wait_parity _
    | Ptx_ir.Mbarrier_slot _
    | Ptx_ir.Tma_load_2d _ | Ptx_ir.Fence_proxy_async
    | Ptx_ir.Wgmma_fence | Ptx_ir.Wgmma_mma_tf32 _
    | Ptx_ir.Gmma_descriptor _
    | Ptx_ir.Wgmma_commit_group | Ptx_ir.Wgmma_wait_group _
    | Ptx_ir.Store_f32 _ | Ptx_ir.Store_f32x2 _ | Ptx_ir.Store_f32x4 _
    | Ptx_ir.Store_f32_masked _ | Ptx_ir.Tile_load_f32_indexed _
    | Ptx_ir.Tile_store_f32_indexed _ | Ptx_ir.Tile_load_f32_2d _
    | Ptx_ir.Tile_store_f32_2d _ | Ptx_ir.Return -> ()) kernel.body;
  let expected_shared_bytes =
    (Ptx_ir.shared_plan kernel.name kernel.body).Ptx_ir.total_bytes in
  require (kernel.launch.shared_bytes = expected_shared_bytes)
    "PTX shared allocation declarations disagree with the launch contract";
  kernel
