(* Every float intrinsic, pinned to one PTX instruction each. Two kernels only
   demonstrate the ones they need (examples/kernels/row_norm.ml), so this
   fixture is what keeps the rest from quietly regressing.

   The Stdlib names lower to the correctly-rounded instruction, so they keep
   their OCaml meaning; the [_approx] names are the special-function unit. *)
open Gpu_dsl

let float_math :
  float gpu_array @ aliased read ->
  float gpu_array @ aliased read_write ->
  unit =
fun x y ->
  let i = Gpu.thread_idx_x () in
  let a = Gpu.load x i in
  let b = Gpu.load x (i +: 1) in
  (* Correctly rounded: sub.rn, div.rn, sqrt.rn, neg, abs, max.NaN, min.NaN. *)
  let exact =
    (a -. b) /. sqrt (abs_float a +. 1.0)
    +. Float.max a b
    +. Float.min a (-. b)
  in
  (* Special-function unit, named for it. *)
  let approx =
    Gpu.exp2_approx a
    +. Gpu.log2_approx (abs_float b +. 1.0)
    +. Gpu.tanh_approx a
    +. Gpu.rsqrt_approx (abs_float b +. 1.0)
    +. Gpu.recip_approx (abs_float a +. 1.0)
  in
  (* The one conversion: an extent becomes a scale factor. *)
  Gpu.store y i ((exact +. approx) *. (1.0 /. float_of_int 256))

(* [Float.max]/[Float.min] claim OCaml's meaning, and the interesting part of
   that claim is the edge cases: OCaml returns nan if either argument is nan,
   and resolves signed zero as max (-0.) (+0.) = +0. PTX without the [.NaN]
   modifier returns the *other* operand when one is nan, which is why the
   emitter asks for [max.NaN.f32]. Thread [i] reduces the pair at [2i] so the
   hardware harness can feed the cases directly. *)
let float_extremes :
  float gpu_array @ aliased read ->
  float gpu_array @ aliased read_write ->
  unit =
fun x y ->
  let i = Gpu.thread_idx_x () in
  let pair = i *: 2 in
  let a = Gpu.load x pair in
  let b = Gpu.load x (pair +: 1) in
  Gpu.store y pair (Float.max a b);
  Gpu.store y (pair +: 1) (Float.min a b)
