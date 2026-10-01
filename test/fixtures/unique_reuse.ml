open Gpu_dsl

let alias_reuse_aliased :
  float gpu_array @ aliased read ->
  float gpu_array @ aliased read_write ->
  float gpu_array @ aliased read_write ->
  unit = fun source target result ->
  let before = Gpu.load source 0 in
  Gpu.store target 0 0.;
  Gpu.store result 0 (before +. Gpu.load source 0)

let unique_reuse :
  float gpu_array @ aliased read ->
  float gpu_array @ unique read_write ->
  float gpu_array @ unique read_write ->
  unit = fun source target result ->
  let before = Gpu.load source 0 in
  Gpu.store target 0 0.;
  Gpu.store result 0 (before +. Gpu.load source 0)
