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
  external ( *. ) : f32 -> f32 -> f32 = "gpu_mul_f32"
end

let saxpy x y a =
  let open F32 in
  let i = Gpu.thread_idx_x () in
  Gpu.store y i
    (a *. Gpu.load x i +. Gpu.load y i)
