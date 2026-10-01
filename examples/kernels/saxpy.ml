open Gpu_dsl

let saxpy :
  float gpu_array @ aliased read ->
  float gpu_array @ unique read_write ->
  float ->
  int ->
  unit = fun x y a n ->
  let i = Gpu.global_idx_x () in
  Gpu.store_masked y i n (a *. Gpu.load_masked x i n +. Gpu.load_masked y i n)
