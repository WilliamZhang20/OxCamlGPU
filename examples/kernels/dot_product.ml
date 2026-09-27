open Gpu_dsl

let dot_product :
  float gpu_array @ aliased read ->
  float gpu_array @ aliased read ->
  float gpu_array @ unique write ->
  unit = fun x y result ->
  let lane = Gpu.thread_idx_x () in
  let product = Gpu.load x lane *. Gpu.load y lane in
  Gpu.store_lane0 result (Gpu.warp_sum_f32 product)
