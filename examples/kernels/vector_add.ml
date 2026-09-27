type f32
type i32
type 'a gpu_array

module Gpu = struct
  external thread_idx_x : unit -> i32 = "gpu_thread_idx_x"
  external load : f32 gpu_array @ aliased read -> i32 -> f32 = "gpu_load_f32"
  external store : f32 gpu_array @ aliased read_write -> i32 -> f32 -> unit = "gpu_store_f32"
end

module F32 = struct
  external ( +. ) : f32 -> f32 -> f32 = "gpu_add_f32"
end

let vector_add x y z =
  let open F32 in
  let i = Gpu.thread_idx_x () in
  Gpu.store z i (Gpu.load x i +. Gpu.load y i)
