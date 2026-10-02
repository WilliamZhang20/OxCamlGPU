open Gpu_dsl
let reject_overflow (y : float gpu_array) n =
  let i = n + 1 in Gpu.store y i 1.
