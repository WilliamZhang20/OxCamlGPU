type 'a gpu_array = 'a array
type 'a gpu_shared = 'a array
type gpu_mbarrier = int array
(* Host stub only — device lowering never materializes this. *)
type tensor_map = int64
type wgmma_acc = float array

module Gpu = struct
  external thread_idx_x : unit -> int = "gpu_thread_idx_x"
  external global_idx_x : unit -> int = "gpu_global_idx_x"
  external block_idx_x : unit -> int = "gpu_block_idx_x"
  external block_idx_y : unit -> int = "gpu_block_idx_y"
  external warpgroup_index : unit -> int = "gpu_warpgroup_index"
  external grouped_tile_m : group:int -> tiles_m:int -> tiles_n:int -> int
    = "gpu_grouped_tile_m"
  external grouped_tile_n : group:int -> tiles_m:int -> tiles_n:int -> int
    = "gpu_grouped_tile_n"
  external load : float gpu_array @ aliased read -> int -> float = "gpu_load_f32"
  external store : float gpu_array @ aliased read_write -> int -> float -> unit = "gpu_store_f32"
  external load_masked : float gpu_array @ aliased read -> int -> int -> float = "gpu_load_f32_masked"
  external store_masked : float gpu_array @ aliased read_write -> int -> int -> float -> unit = "gpu_store_f32_masked"

  external load_f32x4 :
    float gpu_array @ aliased read -> int -> float * float * float * float
    = "gpu_load_f32x4"
  external store_f32x4 :
    float gpu_array @ aliased read_write ->
    int -> float -> float -> float -> float -> unit
    = "gpu_store_f32x4"

  external shared : rows:int -> cols:int -> unit -> float gpu_shared = "gpu_shared_alloc_f32"
  external shared_wgmma : rows:int -> cols:int -> unit -> float gpu_shared = "gpu_shared_alloc_wgmma_f32"
  external shared_load : float gpu_shared -> int -> float = "gpu_shared_load_f32"
  external shared_load_f32x4 :
    float gpu_shared -> int -> float * float * float * float
    = "gpu_shared_load_f32x4"
  external shared_store : float gpu_shared -> int -> float -> unit = "gpu_shared_store_f32"
  external shared_store_f32x4 :
    float gpu_shared -> int -> float -> float -> float -> float -> unit
    = "gpu_shared_store_f32x4"

  external barrier_cta : unit -> unit = "gpu_barrier_cta"
  external cp_async_f32x4 :
    float gpu_shared -> int -> float gpu_array @ aliased read -> int -> unit
    = "gpu_cp_async_f32x4"
  external cp_async_commit : unit -> unit = "gpu_cp_async_commit"
  external cp_async_wait : unit -> unit = "gpu_cp_async_wait"
  external mad_f32 : float -> float -> float -> float = "gpu_mad_f32"

  external warp_sum_f32 : float -> float = "gpu_warp_sum_f32"
  external store_grid_leader : float gpu_array @ unique write -> float -> unit = "gpu_store_grid_leader_f32"

  external mbarrier : unit -> gpu_mbarrier = "gpu_mbarrier_alloc"
  external mbarrier_set : int -> gpu_mbarrier = "gpu_mbarrier_set_alloc"
  external mbarrier_slot : gpu_mbarrier -> int -> gpu_mbarrier
    = "gpu_mbarrier_slot"
  external mbarrier_init : gpu_mbarrier -> int -> unit = "gpu_mbarrier_init"
  external mbarrier_init_elect : gpu_mbarrier -> int -> bool -> unit = "gpu_mbarrier_init_elect"
  external mbarrier_arrive_expect_tx : gpu_mbarrier -> int -> unit = "gpu_mbarrier_arrive_expect_tx"
  external mbarrier_arrive_expect_tx_elect :
    gpu_mbarrier -> int -> bool -> unit = "gpu_mbarrier_arrive_expect_tx_elect"
  external mbarrier_arrive : gpu_mbarrier -> unit = "gpu_mbarrier_arrive"
  external mbarrier_arrive_elect : gpu_mbarrier -> bool -> unit = "gpu_mbarrier_arrive_elect"
  external mbarrier_try_wait_parity : gpu_mbarrier -> int -> unit = "gpu_mbarrier_try_wait_parity"

  external tma_load_2d :
    float gpu_shared -> int -> tensor_map @ aliased read ->
    int -> int -> gpu_mbarrier -> unit = "gpu_tma_load_2d"
  external tma_load_2d_elect :
    float gpu_shared -> int -> tensor_map @ aliased read ->
    int -> int -> gpu_mbarrier -> bool -> unit = "gpu_tma_load_2d_elect"
  external fence_proxy_async : unit -> unit = "gpu_fence_proxy_async"

  external wgmma_fence : unit -> unit = "gpu_wgmma_fence"
  external wgmma_acc : n:int -> unit -> wgmma_acc = "gpu_wgmma_acc"
  external wgmma_acc_get : wgmma_acc -> int -> float = "gpu_wgmma_acc_get"
  external wgmma_acc_row : wgmma_acc -> int -> int = "gpu_wgmma_acc_row"
  external wgmma_acc_col : wgmma_acc -> int -> int = "gpu_wgmma_acc_col"
  external gmma_descriptor : float gpu_shared -> int -> int64
    = "gpu_gmma_descriptor"
  external wgmma_mma_tf32 :
    wgmma_acc -> int64 -> int64 -> n:int -> scale:bool -> unit
    = "gpu_wgmma_mma_tf32"
  external wgmma_commit_group : unit -> unit = "gpu_wgmma_commit_group"
  external wgmma_wait_group : int -> unit = "gpu_wgmma_wait_group"
end
