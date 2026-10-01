open Gpu_dsl

let vector_add :
  float gpu_array @ aliased read ->
  float gpu_array @ aliased read ->
  float gpu_array @ unique read_write ->
  int ->
  unit = fun x y z n ->
  let i = Gpu.global_idx_x () in
  Gpu.store_masked z i n (Gpu.load_masked x i n +. Gpu.load_masked y i n)
