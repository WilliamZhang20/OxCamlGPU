(* Semantic GPU IR: typed SSA values, instructions, and kernels. Every
   constructor here is matched on across the frontend, verifier, transforms and
   backend, so there is nothing to abstract and this module has no signature;
   see docs/architecture.md. *)
open Gpu_type
open Gpu_mode

type value = {
  id : int;
  ty : ty;
  loc : Source_span.t;
  ownership : ownership;
  locality : locality;
  domain_portability : domain_portability;
  gpu_boundary : gpu_boundary;
  permission : permission;
  layout : Layout.t option;
  (* Root buffer identity is provenance for alias analysis. It is not an
     exclusivity claim about this particular derived pointer. *)
  provenance : int option;
}

(** Packed Hopper GMMA shared-memory descriptor fields.
    [leading] and [stride] are in units of 16 bytes;
    [layout_type] occupies bits[63:62] of the descriptor. *)
type gmma_fields = {
  leading : int;
  stride : int;
  layout_type : int;
}

type instr =
  (* Constants, arithmetic, compare, control *)
  | Const_bool of value * bool
  | Sub_i32 of value * value * value
  | Mul_i32 of value * value * value
  | Compare of value * comparison * value * value
  | If of value option * value * region * region
  (* Warp/warpgroup-uniform branch — lowers to bra.uni. *)
  | If_uni of value option * value * region * region
  | Const_i32 of value * int
  | Const_f32 of value * float
  | Add_i32 of value * value * value
  | Add_f32 of value * value * value
  | Mul_f32 of value * value * value
  | Thread_idx_x of value
  | Global_idx_x of value
  | Block_idx_x of value
  | Block_idx_y of value
  | Div_i32 of value * value * value
  | Min_i32 of value * value * value
  | Rem_i32 of value * value * value
  | Mad_f32 of value * value * value * value
  (* Addressing and global memory *)
  | Gep_f32 of value * value * value
  | Load_f32 of value * value
  | Load_f32_masked of value * value * value * value
  (* Aligned vector load: four consecutive f32 from ptr. *)
  | Load_f32x4 of value * value * value * value * value
  | Store_f32 of value * value
  | Store_f32_masked of value * value * value * value
  (* Aligned vector stores: two or four consecutive f32 to ptr. *)
  | Store_f32x2 of value * value * value
  | Store_f32x4 of value * value * value * value * value
  (* Async copy, mbarrier, TMA, WGMMA *)
  | Cp_async_shared_f32x4 of value * value * value
  | Cp_async_commit
  | Cp_async_wait_all
  (* Shared mbarrier (opaque object). Optional predicate: elect-only issue. *)
  | Mbarrier_init of value * value * value option
  | Mbarrier_arrive_expect_tx of value * value * value option
  | Mbarrier_arrive of value * value option
  | Mbarrier_try_wait_parity of value * value
  (* One barrier out of a set, at a dynamic index. The result names the same
     shared storage as the set, offset by the index, so a pipelined schedule
     can address its stage barrier without a static ladder. *)
  | Mbarrier_slot of value * value * value
  (* TMA 2D tile: shared dst, element index, tensormap, (c0,c1), mbarrier. *)
  | Tma_load_2d of value * value * value * value * value * value * value option
  | Fence_proxy_async
  | Wgmma_fence
  (* Pack a GMMA shared descriptor ([dst] : u64). Fields are producer policy. *)
  | Gmma_descriptor of value * value * value * gmma_fields
  (* In-place [acc] += A_desc @ B_desc (TF32 inputs, f32 accumulate). *)
  | Wgmma_mma_tf32 of value list * value * value * int * bool
  | Wgmma_commit_group
  | Wgmma_wait_group of int
  (* Shared memory *)
  | Shared_alloc of value
  | Shared_load_f32 of value * value * value
  (* Four consecutive shared loads starting at index. *)
  | Shared_load_f32x4 of value * value * value * value * value * value
  | Shared_store_f32 of value * value * value
  (* Four consecutive shared stores starting at index. *)
  | Shared_store_f32x4 of value * value * value * value * value * value
  (* Tensors / collectives *)
  | Load_tensor of value * value
  | Store_tensor of value * value
  (* Masked tile movement: runtime nonnegative row/col bounds. Out-of-range
     elements load as 0.0 and stores are skipped. *)
  | Load_tensor_masked of value * value * value * value
  | Store_tensor_masked of value * value * value * value
  (* Change register/lane ownership; shape and dtype must match. *)
  | Remap_tensor of value * value
  | Scale_tensor_f32 of value * value * value
  (* Semantic operations retained until target strategy selection. *)
  | Mul_tensor_f32 of value * value * value
  | Reduce_sum_f32 of value * value
  | Warp_sum_f32 of value * value
  | Store_f32_grid_leader of value * value
  | Barrier of Execution.level
  | Return of value option
  (* Counted loop: induction starts at its current value; while induction < limit,
     run body then induction := induction + step. Body SSA is scoped to the loop. *)
  | For of value * value * int * instr list

and region = { body : instr list; yield : value option }

type arg = { name : string; value : value }
type kernel = {
  name : string;
  args : arg list;
  body : instr list;
  (* Optional CTA width hint for PTX .reqntid. *)
  threads_per_cta : int option;
}
type actual = {
  (* Identifies the complete backing allocation. Distinct ids are a caller
     assertion that the allocations do not overlap; the raw CUDA ABI cannot
     verify ranges because it currently passes pointers without extents. *)
  buffer_id : int;
  actual_ownership : ownership;
  actual_permission : permission;
  actual_locality : locality;
  actual_domain_portability : domain_portability;
  actual_gpu_boundary : gpu_boundary;
}

let make_value ?(loc=Source_span.synthetic) ?(ownership=Aliased) ?(locality=Gpu_mode.Local)
    ?(domain_portability=Domain_portability_unspecified)
    ?(gpu_boundary=Boundary_unspecified) ?(permission=Read_only) ?layout
    ?(provenance=None) id ty =
  { id; ty; loc; ownership; locality; domain_portability; gpu_boundary; permission; layout; provenance }
let results = function
  | Const_bool(v,_) | Sub_i32(v,_,_) | Mul_i32(v,_,_) | Compare(v,_,_,_) -> [v]
  | If(dst,_,_,_) | If_uni(dst,_,_,_) -> Option.to_list dst
  | Const_i32 (v, _) | Const_f32 (v, _) | Thread_idx_x v | Global_idx_x v
  | Block_idx_x v | Block_idx_y v | Gep_f32 (v, _, _) -> [v]
  | Add_i32 (v, _, _) | Add_f32 (v, _, _) | Mul_f32 (v, _, _) | Mad_f32 (v, _, _, _) | Div_i32 (v, _, _) | Rem_i32 (v, _, _) | Min_i32 (v, _, _)
  | Load_f32 (v, _) | Load_f32_masked (v, _, _, _)
  | Shared_alloc v | Shared_load_f32 (v, _, _) | Load_tensor (v, _) | Load_tensor_masked (v, _, _, _) -> [v]
  | Load_f32x4 (a, b, c, d, _) | Shared_load_f32x4 (a, b, c, d, _, _) -> [a; b; c; d]
  | Remap_tensor (v, _) -> [v]
  | Scale_tensor_f32 (v, _, _) -> [v]
  | Mul_tensor_f32 (v, _, _) -> [v]
  | Reduce_sum_f32 (v, _) -> [v]
  | Mbarrier_slot (v, _, _) -> [v]
  | Warp_sum_f32 (v, _) -> [v]
  | Store_f32 _ | Store_f32_masked _ | Store_f32_grid_leader _
  | Store_f32x2 _ | Store_f32x4 _
  | Shared_store_f32 _ | Shared_store_f32x4 _ | Store_tensor _ | Store_tensor_masked _ | Barrier _
  | Cp_async_shared_f32x4 _ | Cp_async_commit | Cp_async_wait_all
  | Mbarrier_init _ | Mbarrier_arrive_expect_tx _ | Mbarrier_arrive _
  | Mbarrier_try_wait_parity _
  | Tma_load_2d _ | Fence_proxy_async
  | Wgmma_fence | Wgmma_mma_tf32 _ | Wgmma_commit_group | Wgmma_wait_group _
  | Return _ | For _ -> []
  | Gmma_descriptor (dst, _, _, _) -> [dst]
let operands = function
  | Const_bool _ -> []
  | Sub_i32(_,a,b) | Mul_i32(_,a,b) | Compare(_,_,a,b)
  | Div_i32(_,a,b) | Rem_i32(_,a,b) | Min_i32(_,a,b) -> [a;b]
  | If(_,c,_,_) | If_uni(_,c,_,_) -> [c]
  | Const_i32 _ | Const_f32 _ | Thread_idx_x _ | Global_idx_x _ | Block_idx_x _ | Block_idx_y _ -> []
  | Gep_f32 (_, p, ix) -> [p; ix]
  | Add_i32 (_, a, b) | Add_f32 (_, a, b) | Mul_f32 (_, a, b) -> [a; b]
  | Mad_f32 (_, a, b, c) -> [a; b; c]
  | Load_f32 (_, p) -> [p] | Load_f32_masked (_, p, ix, n) -> [p; ix; n]
  | Load_f32x4 (_, _, _, _, p) -> [p]
  | Store_f32 (p, v) -> [p; v] | Store_f32_masked (p, v, ix, n) -> [p; v; ix; n]
  | Store_f32x2 (p, a, b) -> [p; a; b]
  | Store_f32x4 (p, a, b, c, d) -> [p; a; b; c; d]
  | Shared_alloc _ | Barrier _ | Fence_proxy_async -> []
  | Shared_load_f32 (_, p, ix) -> [p; ix]
  | Shared_load_f32x4 (_, _, _, _, p, ix) -> [p; ix]
  | Shared_store_f32 (p, v, ix) -> [p; v; ix]
  | Shared_store_f32x4 (p, a, b, c, d, ix) -> [p; a; b; c; d; ix]
  | Cp_async_shared_f32x4 (memref, index, ptr) -> [memref; index; ptr]
  | Cp_async_commit | Cp_async_wait_all -> []
  | Mbarrier_init (mbar, count, pred) ->
      mbar :: count :: Option.to_list pred
  | Mbarrier_arrive_expect_tx (mbar, tx, pred) ->
      mbar :: tx :: Option.to_list pred
  | Mbarrier_arrive (mbar, pred) -> mbar :: Option.to_list pred
  | Mbarrier_try_wait_parity (mbar, parity) -> [mbar; parity]
  | Mbarrier_slot (_, set, index) -> [set; index]
  | Tma_load_2d (smem, index, tmap, c0, c1, mbar, pred) ->
      smem :: index :: tmap :: c0 :: c1 :: mbar :: Option.to_list pred
  | Wgmma_fence | Wgmma_commit_group | Wgmma_wait_group _ -> []
  | Gmma_descriptor (_, smem, index, _) -> [smem; index]
  | Wgmma_mma_tf32 (acc, desc_a, desc_b, _, _) ->
      acc @ [desc_a; desc_b]
  | Load_tensor (_, src) -> [src]
  | Load_tensor_masked (_, src, rows, cols) -> [src; rows; cols]
  | Remap_tensor (_, src) -> [src]
  | Scale_tensor_f32 (_, tensor, scalar) -> [tensor; scalar]
  | Mul_tensor_f32 (_, a, b) -> [a; b]
  | Reduce_sum_f32 (_, tensor) -> [tensor]
  | Warp_sum_f32 (_, value) -> [value]
  | Store_f32_grid_leader (ptr, value) -> [ptr; value]
  | Store_tensor (dst, src) -> [dst; src]
  | Store_tensor_masked (dst, src, rows, cols) -> [dst; src; rows; cols]
  | Return None -> [] | Return (Some v) -> [v]
  | For (induction, limit, _, _) -> [induction; limit]

(* Rebuild an instruction with transformed operand values. Definitions remain
   unchanged; nested regions and their yields are traversed as well. *)
let rec map_uses replace = function
  | Const_bool _ as i -> i
  | Sub_i32 (d, a, b) -> Sub_i32(d,replace a,replace b)
  | Mul_i32 (d, a, b) -> Mul_i32(d,replace a,replace b)
  | Compare (d, c, a, b) -> Compare(d,c,replace a,replace b)
  | If (d, c, a, b) ->
      let region (r : region) =
        { body = List.map (map_uses replace) r.body;
          yield = Option.map replace r.yield }
      in
      If(d,replace c,region a,region b)
  | If_uni (d, c, a, b) ->
      let region (r : region) =
        { body = List.map (map_uses replace) r.body;
          yield = Option.map replace r.yield }
      in
      If_uni(d,replace c,region a,region b)
  | (Const_i32 _ | Const_f32 _ | Thread_idx_x _ | Global_idx_x _ |
     Block_idx_x _ | Block_idx_y _ | Shared_alloc _ | Return None) as i -> i
  | Add_i32 (dst,a,b) -> Add_i32(dst,replace a,replace b)
  | Add_f32 (dst,a,b) -> Add_f32(dst,replace a,replace b)
  | Mul_f32 (dst,a,b) -> Mul_f32(dst,replace a,replace b)
  | Mad_f32 (dst,a,b,c) -> Mad_f32(replace dst,replace a,replace b,replace c)
  | Div_i32 (dst,a,b) -> Div_i32(dst,replace a,replace b)
  | Min_i32 (dst,a,b) -> Min_i32(dst,replace a,replace b)
  | Rem_i32 (dst,a,b) -> Rem_i32(dst,replace a,replace b)
  | Gep_f32 (dst,base,index) -> Gep_f32(dst,replace base,replace index)
  | Load_f32 (dst,ptr) -> Load_f32(dst,replace ptr)
  | Load_f32_masked (dst,ptr,index,bound) ->
      Load_f32_masked(dst,replace ptr,replace index,replace bound)
  | Load_f32x4 (a,b,c,d,ptr) ->
      Load_f32x4(a,b,c,d,replace ptr)
  | Store_f32 (ptr,value) -> Store_f32(replace ptr,replace value)
  | Store_f32_masked (ptr,value,index,bound) ->
      Store_f32_masked(replace ptr,replace value,replace index,replace bound)
  | Store_f32x2 (ptr,a,b) -> Store_f32x2(replace ptr,replace a,replace b)
  | Store_f32x4 (ptr,a,b,c,d) ->
      Store_f32x4(replace ptr,replace a,replace b,replace c,replace d)
  | Shared_load_f32 (dst,memref,index) ->
      Shared_load_f32(dst,replace memref,replace index)
  | Shared_load_f32x4 (a,b,c,d,memref,index) ->
      Shared_load_f32x4(a,b,c,d,replace memref,replace index)
  | Shared_store_f32 (memref,value,index) ->
      Shared_store_f32(replace memref,replace value,replace index)
  | Shared_store_f32x4 (memref,a,b,c,d,index) ->
      Shared_store_f32x4(replace memref,replace a,replace b,replace c,replace d,replace index)
  | Cp_async_shared_f32x4 (memref,index,ptr) ->
      Cp_async_shared_f32x4(replace memref,replace index,replace ptr)
  | (Cp_async_commit | Cp_async_wait_all | Fence_proxy_async
    | Wgmma_fence | Wgmma_commit_group) as i -> i
  | Wgmma_wait_group _ as i -> i
  | Mbarrier_init (mbar, count, pred) ->
      Mbarrier_init (replace mbar, replace count, Option.map replace pred)
  | Mbarrier_arrive_expect_tx (mbar, tx, pred) ->
      Mbarrier_arrive_expect_tx (replace mbar, replace tx, Option.map replace pred)
  | Mbarrier_arrive (mbar, pred) ->
      Mbarrier_arrive (replace mbar, Option.map replace pred)
  | Mbarrier_try_wait_parity (mbar, parity) ->
      Mbarrier_try_wait_parity (replace mbar, replace parity)
  | Mbarrier_slot (dst, set, index) ->
      Mbarrier_slot (dst, replace set, replace index)
  | Tma_load_2d (smem, index, tmap, c0, c1, mbar, pred) ->
      Tma_load_2d
        ( replace smem, replace index, replace tmap, replace c0, replace c1
        , replace mbar, Option.map replace pred )
  | Gmma_descriptor (dst, smem, index, fields) ->
      Gmma_descriptor (dst, replace smem, replace index, fields)
  | Wgmma_mma_tf32 (acc, desc_a, desc_b, n, scale_d) ->
      Wgmma_mma_tf32 (List.map replace acc, replace desc_a, replace desc_b, n, scale_d)
  | Load_tensor (dst,src) -> Load_tensor(dst,replace src)
  | Load_tensor_masked (dst,src,rows,cols) ->
      Load_tensor_masked(dst,replace src,replace rows,replace cols)
  | Store_tensor (dst,src) -> Store_tensor(replace dst,replace src)
  | Store_tensor_masked (dst,src,rows,cols) ->
      Store_tensor_masked(replace dst,replace src,replace rows,replace cols)
  | Remap_tensor (dst,src) -> Remap_tensor(dst,replace src)
  | Scale_tensor_f32 (dst,src,scalar) ->
      Scale_tensor_f32(dst,replace src,replace scalar)
  | Mul_tensor_f32 (dst,a,b) -> Mul_tensor_f32(dst,replace a,replace b)
  | Reduce_sum_f32 (dst,src) -> Reduce_sum_f32(dst,replace src)
  | Warp_sum_f32 (dst,src) -> Warp_sum_f32(dst,replace src)
  | Store_f32_grid_leader (ptr,value) ->
      Store_f32_grid_leader(replace ptr,replace value)
  | Barrier _ as i -> i
  | Return (Some value) -> Return (Some (replace value))
  | For (induction, limit, step, body) ->
      For (replace induction, replace limit, step, List.map (map_uses replace) body)

let rec flatten body =
  List.concat_map
    (fun i ->
      i
      :: (match i with
          | If (_, _, a, b) | If_uni (_, _, a, b) -> flatten a.body @ flatten b.body
          | For (_, _, _, body) -> flatten body
          | _ -> []))
    body
