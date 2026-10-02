open Gpu_dsl
let ( +. ) _ _ = 17.
let reject_operator (y : float gpu_array) = Gpu.store y 0 (1. +. 2.)
