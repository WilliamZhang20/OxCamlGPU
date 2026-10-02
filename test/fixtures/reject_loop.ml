open Gpu_dsl
let reject_loop (y : float gpu_array) = for i = 0 to 3 do Gpu.store y i 1. done
