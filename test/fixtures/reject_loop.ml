open Gpu_dsl
(* downto is rejected; ascending for is supported *)
let reject_loop (y : float gpu_array) =
  for i = 3 downto 0 do Gpu.store y i 1. done
