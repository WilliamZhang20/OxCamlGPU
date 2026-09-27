open Gpu_dsl

let vector_add :
  float gpu_array @ aliased read ->
  float gpu_array @ aliased read ->
  float gpu_array @ unique read_write ->
  unit = fun x y z ->
  let i = Gpu.thread_idx_x () in
  Gpu.store z i (Gpu.load x i +. Gpu.load y i)
