type 'a gpu_array = 'a array

module Gpu = struct
  external thread_idx_x : unit -> int = "gpu_thread_idx_x"
  external load : float gpu_array @ aliased read -> int -> float = "gpu_load_f32"
  external store : float gpu_array @ aliased read_write -> int -> float -> unit = "gpu_store_f32"
end
