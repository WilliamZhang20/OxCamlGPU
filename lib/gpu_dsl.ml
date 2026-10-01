type 'a gpu_array = 'a array

module Gpu = struct
  external thread_idx_x : unit -> int = "gpu_thread_idx_x"
  external global_idx_x : unit -> int = "gpu_global_idx_x"
  external load : float gpu_array @ aliased read -> int -> float = "gpu_load_f32"
  external store : float gpu_array @ aliased read_write -> int -> float -> unit = "gpu_store_f32"
  external load_masked : float gpu_array @ aliased read -> int -> int -> float = "gpu_load_f32_masked"
  external store_masked : float gpu_array @ aliased read_write -> int -> int -> float -> unit = "gpu_store_f32_masked"
  external warp_sum_f32 : float -> float = "gpu_warp_sum_f32"
  external store_grid_leader : float gpu_array @ unique write -> float -> unit = "gpu_store_grid_leader_f32"
end
