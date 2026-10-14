(* Target PTX IR: legalized operations prior to physicalization and printing,
   plus the shared-memory window plan the launch contract and the printer both
   read. No semantic Ir.instr is carried across this boundary: unsupported
   semantic operations must fail during lowering. *)
type operation =
  | Const_bool of Ir.value * bool
  | Sub_i32 of Ir.value * Ir.value * Ir.value
  | Mul_i32 of Ir.value * Ir.value * Ir.value
  | Div_i32 of Ir.value * Ir.value * Ir.value
  | Min_i32 of Ir.value * Ir.value * Ir.value
  | Rem_i32 of Ir.value * Ir.value * Ir.value
  | Compare of Ir.value * Gpu_type.comparison * Ir.value * Ir.value
  | Label of int
  | Jump of int
  | Jump_uni of int
  | Branch of Ir.value * int * int
  | Branch_uni of Ir.value * int * int
  | Move of Ir.value * Ir.value
  | Const_i32 of Ir.value * int
  | Const_f32 of Ir.value * float
  | Add_i32 of Ir.value * Ir.value * Ir.value
  | Add_f32 of Ir.value * Ir.value * Ir.value
  | Mul_f32 of Ir.value * Ir.value * Ir.value
  (* dst := a * b + c *)
  | Mad_f32 of Ir.value * Ir.value * Ir.value * Ir.value
  | Unary_f32 of Ir.value * Gpu_type.unary_f32 * Ir.value
  | Binary_f32 of Ir.value * Gpu_type.binary_f32 * Ir.value * Ir.value
  | F32_of_i32 of Ir.value * Ir.value
  | Tensor_scale_f32 of Ir.value * Ir.value * Ir.value
  | Tensor_mul_f32 of Ir.value * Ir.value * Ir.value
  | Thread_idx_x of Ir.value
  | Global_idx_x of Ir.value
  | Block_idx_x of Ir.value
  | Block_idx_y of Ir.value
  | Gep_f32 of Ir.value * Ir.value * Ir.value
  | Load_f32 of Ir.value * Ir.value
  | Load_f32_masked of Ir.value * Ir.value * Ir.value * Ir.value
  | Load_f32x4 of Ir.value * Ir.value * Ir.value * Ir.value * Ir.value
  | Store_f32 of Ir.value * Ir.value
  | Store_f32_masked of Ir.value * Ir.value * Ir.value * Ir.value
  | Store_f32x2 of Ir.value * Ir.value * Ir.value
  | Store_f32x4 of Ir.value * Ir.value * Ir.value * Ir.value * Ir.value
  | Shared_alloc of Ir.value
  | Shared_load_f32 of Ir.value * Ir.value * Ir.value
  | Shared_load_f32x4 of Ir.value * Ir.value * Ir.value * Ir.value * Ir.value * Ir.value
  | Shared_store_f32 of Ir.value * Ir.value * Ir.value
  | Shared_store_f32x4 of Ir.value * Ir.value * Ir.value * Ir.value * Ir.value * Ir.value
  | Cp_async_shared_f32x4 of Ir.value * Ir.value * Ir.value
  | Cp_async_commit
  | Cp_async_wait_all
  | Mbarrier_init of Ir.value * Ir.value * Ir.value option
  | Mbarrier_arrive_expect_tx of Ir.value * Ir.value * Ir.value option
  | Mbarrier_arrive of Ir.value * Ir.value option
  | Mbarrier_try_wait_parity of Ir.value * Ir.value
  | Mbarrier_slot of Ir.value * Ir.value * Ir.value
  | Tma_load_2d of Ir.value * Ir.value * Ir.value * Ir.value * Ir.value * Ir.value
      * Ir.value option
  | Fence_proxy_async
  | Wgmma_fence
  | Gmma_descriptor of Ir.value * Ir.value * Ir.value * Ir.gmma_fields
  | Wgmma_mma of Gpu_type.dtype * Ir.value list * Ir.value * Ir.value * int * bool
  | Wgmma_commit_group
  | Wgmma_wait_group of int
  | Tile_load_f32 of Ir.value * Ir.value
  | Tile_store_f32 of Ir.value * Ir.value
  | Tile_load_f32_indexed of Ir.value * Ir.value * int * int
  | Tile_store_f32_indexed of Ir.value * Ir.value * int * int
  | Tile_load_f32_masked of Ir.value * Ir.value * Ir.value * Ir.value
  | Tile_store_f32_masked of Ir.value * Ir.value * Ir.value * Ir.value
  (* tile_cols, row_offset, col_offset, row_stride; optional runtime row/col bounds *)
  | Tile_load_f32_2d of Ir.value * Ir.value * int * int * int * int * (Ir.value * Ir.value) option
  | Tile_store_f32_2d of Ir.value * Ir.value * int * int * int * int * (Ir.value * Ir.value) option
  | Store_grid_leader_f32 of Ir.value * Ir.value
  | Barrier_cta
  | Tensor_lane_f32 of Ir.value * Ir.value
  | Warp_butterfly_sum_f32 of Ir.value * Ir.value
  | Return

(* --- shared memory window ---------------------------------------------- *)

(* All of a kernel's shared allocations live in one window so that a single
   declaration decides static vs dynamic placement. sm_90 caps a *statically*
   declared .shared array at 48 KiB; the 228 KiB an SM really has is only
   reachable through the dynamic (.extern .shared) window, which the launch
   must request. Kernels that fit stay static so their launchers keep passing
   zero dynamic bytes. *)
let static_shared_limit = 49152
let max_shared_per_cta = 232448
let shared_window_symbol = "__oxgpu_smem"
let shared_window_align = 128

type shared_plan = {
  (* Byte offset of each shared allocation inside the window, by MemRef id. *)
  offsets : (int * int) list;
  total_bytes : int;
  (* Dynamic bytes a launch must request: 0 when the window is static. *)
  dynamic_bytes : int;
}

(* [shared_plan name body] lays out [body]'s shared allocations by ascending
   MemRef id, so the same kernel always yields the same offsets. [name] only
   appears in the over-budget error. *)
let shared_plan name body =
  let allocation_bytes (m : Ir.value) =
    match m.Ir.ty, m.Ir.layout with
    | Gpu_type.MemRef (shape, dtype, Gpu_type.Shared), Some layout ->
        Layout.shared_storage_bytes shape dtype layout
    | _ -> Gpu_type.storage_bytes_of_ty m.Ir.ty
  in
  let allocations =
    List.filter_map
      (function Shared_alloc m -> Some m | _ -> None) body
  in
  (* Lay out by ascending id so the same kernel always yields the same PTX. *)
  let allocations =
    List.sort_uniq (fun a b -> Int.compare a.Ir.id b.Ir.id) allocations in
  let align n = ((n + shared_window_align - 1) / shared_window_align)
                * shared_window_align in
  (* Each allocation starts on the window alignment, which GMMA descriptors
     and TMA both need. The window ends with the last allocation rather than
     on an alignment boundary, so a single allocation costs exactly its own
     size. *)
  let offsets, total =
    List.fold_left (fun (offsets, next) m ->
      let size = allocation_bytes m in
      if size < 0 then invalid_arg "shared allocation size is negative";
      let start = align next in
      ((m.Ir.id, start) :: offsets, start + size)) ([], 0) allocations
  in
  if total > max_shared_per_cta then
    invalid_arg
      (Printf.sprintf
         "kernel %s needs %d bytes of shared memory; an SM provides %d"
         name total max_shared_per_cta);
  { offsets = List.rev offsets;
    total_bytes = total;
    dynamic_bytes = (if total > static_shared_limit then total else 0) }


type launch_contract = {
  threads_per_cta : (int * int * int) option;
  (* Size of the kernel's one shared window; dynamic above the static cap. *)
  shared_bytes : int;
}

type kernel = {
  name : string;
  args : Ir.arg list;
  body : operation list;
  launch : launch_contract;
}

let results = function
  | Const_bool(v,_) | Sub_i32(v,_,_) | Mul_i32(v,_,_) | Div_i32(v,_,_) | Rem_i32(v,_,_)
  | Min_i32(v,_,_)
  | Compare(v,_,_,_) | Move(v,_) | Shared_alloc v -> [v]
  | Label _ | Jump _ | Jump_uni _ | Branch _ | Branch_uni _ -> []
  | Const_i32 (v, _) | Const_f32 (v, _) | Thread_idx_x v | Global_idx_x v
  | Block_idx_x v | Block_idx_y v
  | Gep_f32 (v, _, _) | Load_f32 (v, _) | Load_f32_masked (v, _, _, _)
  | Shared_load_f32 (v, _, _)
  | Tile_load_f32 (v, _) | Tile_load_f32_indexed (v, _, _, _)
  | Tile_load_f32_masked (v, _, _, _) | Tile_load_f32_2d (v, _, _, _, _, _, _)
  | Tensor_lane_f32 (v, _) | Warp_butterfly_sum_f32 (v, _) -> [v]
  | Load_f32x4 (a, b, c, d, _) | Shared_load_f32x4 (a, b, c, d, _, _) -> [a; b; c; d]
  | Add_i32 (v, _, _) | Add_f32 (v, _, _) | Mul_f32 (v, _, _) | Mad_f32 (v, _, _, _)
  | Unary_f32 (v, _, _) | Binary_f32 (v, _, _, _) | F32_of_i32 (v, _)
  | Tensor_scale_f32 (v, _, _) | Tensor_mul_f32 (v, _, _) -> [v]
  | Mbarrier_slot (v, _, _) -> [v]
  | Store_f32 _ | Store_f32_masked _ | Store_f32x2 _ | Store_f32x4 _
  | Shared_store_f32 _ | Shared_store_f32x4 _
  | Cp_async_shared_f32x4 _ | Cp_async_commit | Cp_async_wait_all
  | Mbarrier_init _ | Mbarrier_arrive_expect_tx _ | Mbarrier_arrive _
  | Mbarrier_try_wait_parity _
  | Tma_load_2d _ | Fence_proxy_async
  | Wgmma_fence | Wgmma_mma _ | Wgmma_commit_group | Wgmma_wait_group _
  | Tile_store_f32 _ | Tile_store_f32_indexed _
  | Tile_store_f32_masked _ | Tile_store_f32_2d _
  | Store_grid_leader_f32 _ | Barrier_cta | Return -> []
  | Gmma_descriptor (dst, _, _, _) -> [dst]

let operands = function
  | Const_bool _ | Label _ | Jump _ | Jump_uni _ -> []
  | Sub_i32(_,a,b) | Mul_i32(_,a,b) | Div_i32(_,a,b) | Rem_i32(_,a,b) | Min_i32(_,a,b)
  | Compare(_,_,a,b) -> [a;b]
  | Branch(v,_,_) | Branch_uni(v,_,_) | Move(_,v) -> [v]
  | Const_i32 _ | Const_f32 _ | Thread_idx_x _ | Global_idx_x _ | Block_idx_x _ | Block_idx_y _
  | Shared_alloc _ | Barrier_cta | Return -> []
  | Add_i32 (_, a, b) | Add_f32 (_, a, b) | Mul_f32 (_, a, b)
  | Tensor_scale_f32 (_, a, b) | Tensor_mul_f32 (_, a, b) -> [a;b]
  | Mad_f32 (_, a, b, c) -> [a; b; c]
  | Unary_f32 (_, _, a) | F32_of_i32 (_, a) -> [a]
  | Binary_f32 (_, _, a, b) -> [a; b]
  | Gep_f32 (_, p, i) -> [p;i]
  | Load_f32 (_, p) | Tile_load_f32 (_, p) | Tensor_lane_f32 (_, p)
  | Warp_butterfly_sum_f32 (_, p) -> [p]
  | Load_f32x4 (_, _, _, _, p) -> [p]
  | Shared_load_f32 (_, p, i) -> [p;i]
  | Shared_load_f32x4 (_, _, _, _, p, i) -> [p; i]
  | Shared_store_f32 (p, v, i) -> [p;v;i]
  | Shared_store_f32x4 (p, a, b, c, d, i) -> [p; a; b; c; d; i]
  | Cp_async_shared_f32x4 (p, i, g) -> [p; i; g]
  | Cp_async_commit | Cp_async_wait_all | Fence_proxy_async
  | Wgmma_fence | Wgmma_commit_group | Wgmma_wait_group _ -> []
  | Mbarrier_init (m, c, p) -> m :: c :: Option.to_list p
  | Mbarrier_arrive_expect_tx (m, t, p) -> m :: t :: Option.to_list p
  | Mbarrier_arrive (m, p) -> m :: Option.to_list p
  | Mbarrier_try_wait_parity (m, p) -> [m; p]
  | Mbarrier_slot (_, set, index) -> [set; index]
  | Tma_load_2d (s, i, t, c0, c1, m, p) ->
      s :: i :: t :: c0 :: c1 :: m :: Option.to_list p
  | Gmma_descriptor (_, smem, index, _) -> [smem; index]
  | Wgmma_mma (_, acc, desc_a, desc_b, _, _) ->
      acc @ [desc_a; desc_b]
  | Tile_load_f32_indexed (_, p, _, _) -> [p]
  | Tile_load_f32_masked (_, p, rows, cols) -> [p;rows;cols]
  | Tile_load_f32_2d (_, p, _, _, _, _, bounds) ->
      p :: (match bounds with None -> [] | Some (rows, cols) -> [rows; cols])
  | Load_f32_masked (_, p, i, n) -> [p;i;n]
  | Store_f32 (p, v) | Tile_store_f32 (p, v) | Store_grid_leader_f32 (p, v) -> [p;v]
  | Store_f32x2 (p, a, b) -> [p; a; b]
  | Store_f32x4 (p, a, b, c, d) -> [p; a; b; c; d]
  | Tile_store_f32_indexed (p, v, _, _) -> [p;v]
  | Tile_store_f32_masked (p, v, rows, cols) -> [p;v;rows;cols]
  | Tile_store_f32_2d (p, v, _, _, _, _, bounds) ->
      p :: v :: (match bounds with None -> [] | Some (rows, cols) -> [rows; cols])
  | Store_f32_masked (p, v, i, n) -> [p;v;i;n]

(* Rebuild an operation with [f] applied to every value it names, results and
   operands alike. Labels, immediates and shapes are untouched. *)
let map_values f = function
  | Const_bool (d, b) -> Const_bool (f d, b)
  | Sub_i32 (d, a, b) -> Sub_i32 (f d, f a, f b)
  | Mul_i32 (d, a, b) -> Mul_i32 (f d, f a, f b)
  | Div_i32 (d, a, b) -> Div_i32 (f d, f a, f b)
  | Min_i32 (d, a, b) -> Min_i32 (f d, f a, f b)
  | Rem_i32 (d, a, b) -> Rem_i32 (f d, f a, f b)
  | Compare (d, c, a, b) -> Compare (f d, c, f a, f b)
  | Label _ | Jump _ | Jump_uni _ as op -> op
  | Branch (v, yes, no) -> Branch (f v, yes, no)
  | Branch_uni (v, yes, no) -> Branch_uni (f v, yes, no)
  | Move (d, s) -> Move (f d, f s)
  | Const_i32 (d, n) -> Const_i32 (f d, n)
  | Const_f32 (d, x) -> Const_f32 (f d, x)
  | Add_i32 (d, a, b) -> Add_i32 (f d, f a, f b)
  | Add_f32 (d, a, b) -> Add_f32 (f d, f a, f b)
  | Mul_f32 (d, a, b) -> Mul_f32 (f d, f a, f b)
  | Mad_f32 (d, a, b, c) -> Mad_f32 (f d, f a, f b, f c)
  | Unary_f32 (d, op, a) -> Unary_f32 (f d, op, f a)
  | Binary_f32 (d, op, a, b) -> Binary_f32 (f d, op, f a, f b)
  | F32_of_i32 (d, a) -> F32_of_i32 (f d, f a)
  | Tensor_scale_f32 (d, a, b) -> Tensor_scale_f32 (f d, f a, f b)
  | Tensor_mul_f32 (d, a, b) -> Tensor_mul_f32 (f d, f a, f b)
  | Thread_idx_x d -> Thread_idx_x (f d)
  | Global_idx_x d -> Global_idx_x (f d)
  | Block_idx_x d -> Block_idx_x (f d)
  | Block_idx_y d -> Block_idx_y (f d)
  | Gep_f32 (d, p, i) -> Gep_f32 (f d, f p, f i)
  | Load_f32 (d, p) -> Load_f32 (f d, f p)
  | Load_f32_masked (d, p, i, n) -> Load_f32_masked (f d, f p, f i, f n)
  | Load_f32x4 (a, b, c, d, p) -> Load_f32x4 (f a, f b, f c, f d, f p)
  | Store_f32 (p, v) -> Store_f32 (f p, f v)
  | Store_f32_masked (p, v, i, n) -> Store_f32_masked (f p, f v, f i, f n)
  | Store_f32x2 (p, a, b) -> Store_f32x2 (f p, f a, f b)
  | Store_f32x4 (p, a, b, c, d) -> Store_f32x4 (f p, f a, f b, f c, f d)
  | Shared_alloc m -> Shared_alloc (f m)
  | Shared_load_f32 (d, p, i) -> Shared_load_f32 (f d, f p, f i)
  | Shared_load_f32x4 (a, b, c, d, p, i) ->
      Shared_load_f32x4 (f a, f b, f c, f d, f p, f i)
  | Shared_store_f32 (p, v, i) -> Shared_store_f32 (f p, f v, f i)
  | Shared_store_f32x4 (p, a, b, c, d, i) ->
      Shared_store_f32x4 (f p, f a, f b, f c, f d, f i)
  | Cp_async_shared_f32x4 (p, i, g) -> Cp_async_shared_f32x4 (f p, f i, f g)
  | Cp_async_commit | Cp_async_wait_all as op -> op
  | Mbarrier_init (m, c, p) -> Mbarrier_init (f m, f c, Option.map f p)
  | Mbarrier_arrive_expect_tx (m, t, p) ->
      Mbarrier_arrive_expect_tx (f m, f t, Option.map f p)
  | Mbarrier_arrive (m, p) -> Mbarrier_arrive (f m, Option.map f p)
  | Mbarrier_try_wait_parity (m, p) -> Mbarrier_try_wait_parity (f m, f p)
  | Mbarrier_slot (d, set, i) -> Mbarrier_slot (f d, f set, f i)
  | Tma_load_2d (s, i, t, c0, c1, m, p) ->
      Tma_load_2d (f s, f i, f t, f c0, f c1, f m, Option.map f p)
  | Fence_proxy_async | Wgmma_fence | Wgmma_commit_group as op -> op
  | Gmma_descriptor (d, s, i, fields) -> Gmma_descriptor (f d, f s, f i, fields)
  | Wgmma_mma (element, acc, da, db, n, scale) ->
      Wgmma_mma (element, List.map f acc, f da, f db, n, scale)
  | Wgmma_wait_group _ as op -> op
  | Tile_load_f32 (t, m) -> Tile_load_f32 (f t, f m)
  | Tile_store_f32 (m, t) -> Tile_store_f32 (f m, f t)
  | Tile_load_f32_indexed (d, p, base, stride) ->
      Tile_load_f32_indexed (f d, f p, base, stride)
  | Tile_store_f32_indexed (p, v, base, stride) ->
      Tile_store_f32_indexed (f p, f v, base, stride)
  | Tile_load_f32_masked (t, m, rows, cols) ->
      Tile_load_f32_masked (f t, f m, f rows, f cols)
  | Tile_store_f32_masked (m, t, rows, cols) ->
      Tile_store_f32_masked (f m, f t, f rows, f cols)
  | Tile_load_f32_2d (d, p, cols, ro, co, rs, bounds) ->
      Tile_load_f32_2d (f d, f p, cols, ro, co, rs,
        Option.map (fun (r, c) -> f r, f c) bounds)
  | Tile_store_f32_2d (p, v, cols, ro, co, rs, bounds) ->
      Tile_store_f32_2d (f p, f v, cols, ro, co, rs,
        Option.map (fun (r, c) -> f r, f c) bounds)
  | Store_grid_leader_f32 (p, v) -> Store_grid_leader_f32 (f p, f v)
  | Barrier_cta -> Barrier_cta
  | Tensor_lane_f32 (d, t) -> Tensor_lane_f32 (f d, f t)
  | Warp_butterfly_sum_f32 (d, s) -> Warp_butterfly_sum_f32 (f d, f s)
  | Return -> Return

let all_values kernel =
  List.concat_map (fun instruction -> operands instruction @ results instruction) kernel.body
  @ List.map (fun arg -> arg.Ir.value) kernel.args
