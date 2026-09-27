open Gpu_dsl

let literal_probe :
  float gpu_array @ aliased read ->
  float gpu_array @ unique read_write ->
  unit = fun input output ->
  let i = Gpu.thread_idx_x () in
  Gpu.store output i (Gpu.load input i +. 2.5)
