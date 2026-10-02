open Gpu_dsl
module Fake = struct
  module Gpu_dsl = struct module Gpu = struct let load _ _ = 42. end end
end
let reject_shadow (x : float gpu_array) (y : float gpu_array) =
  Gpu.store y 0 (Fake.Gpu_dsl.Gpu.load x 0)
