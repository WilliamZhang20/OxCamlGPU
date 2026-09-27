open Gpu_dsl

let saxpy :
  float gpu_array @ aliased read ->
  float gpu_array @ unique read_write ->
  float ->
  unit = fun x y a ->
  let i = Gpu.thread_idx_x () in
  Gpu.store y i (a *. Gpu.load x i +. Gpu.load y i)
