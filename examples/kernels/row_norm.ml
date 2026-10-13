(* Row normalizations: softmax and RMS norm, one CTA per row.

   These are the two shapes that need more than add and multiply. Softmax
   wants a maximum, an exponential and a reciprocal; RMS norm wants a
   reciprocal square root. Each is one PTX instruction, and the [_approx]
   names say which ones run on the special-function unit — choosing the fast
   unit is a numerical decision, so it is written down rather than inferred.

   [reduce] is the recursive tree fold from block_reduce.ml, parameterized by
   its operator and reused at three of them. It is a compile-time function
   value, so it leaves no call behind; see docs/functions.md. *)
open Gpu_dsl

(* Reduce [v] across the CTA and return the result to every thread. [sm] is
   left reusable: the closing barrier is what makes a second reduction into
   the same tile safe. *)
let reduce sm tid ~size ~combine v =
  Gpu.shared_store sm tid v;
  Gpu.barrier_cta ();
  let rec fold stride =
    if stride = 0 then ()
    else begin
      if tid < stride then
        Gpu.shared_store sm tid
          (combine (Gpu.shared_load sm tid)
             (Gpu.shared_load sm (tid +: stride)));
      Gpu.barrier_cta ();
      fold (stride / 2)
    end
  in
  fold (size / 2);
  let total = Gpu.shared_load sm 0 in
  Gpu.barrier_cta ();
  total

(* exp(x) = exp2(x * log2 e). The hardware has ex2, not ex, so the change of
   base stays visible at the one place it happens. *)
let row_softmax ~cols =
  let schedule
      (x : float gpu_array @ aliased read)
      (y : float gpu_array @ unique read_write) =
    let log2_e = 1.4426950408889634 in
    let sm = Gpu.shared ~rows:1 ~cols () in
    let tid = Gpu.thread_idx_x () in
    let base = Gpu.block_idx_x () *: cols in
    let v = Gpu.load x (base +: tid) in
    (* Subtracting the row maximum is what keeps exp in range. *)
    let peak =
      reduce sm tid ~size:cols ~combine:(fun a b -> Float.max a b) v
    in
    let weight = Gpu.exp2_approx ((v -. peak) *. log2_e) in
    let total =
      reduce sm tid ~size:cols ~combine:(fun a b -> a +. b) weight
    in
    Gpu.store y (base +: tid) (weight *. Gpu.recip_approx total)
  in
  schedule

(* y = x / sqrt(mean(x^2) + eps), the normalization transformers actually use.
   rsqrt is one instruction, so there is no divide and no separate sqrt. *)
let row_rms_norm ~cols =
  let schedule
      (x : float gpu_array @ aliased read)
      (y : float gpu_array @ unique read_write) =
    let inv_cols = 1.0 /. float_of_int cols in
    let eps = 1e-6 in
    let sm = Gpu.shared ~rows:1 ~cols () in
    let tid = Gpu.thread_idx_x () in
    let base = Gpu.block_idx_x () *: cols in
    let v = Gpu.load x (base +: tid) in
    let sum_squares =
      reduce sm tid ~size:cols ~combine:(fun a b -> a +. b) (v *. v)
    in
    let scale = Gpu.rsqrt_approx ((sum_squares *. inv_cols) +. eps) in
    Gpu.store y (base +: tid) (v *. scale)
  in
  schedule

let softmax_256 = row_softmax ~cols:256 [@@gpu.threads 256]
let rms_norm_256 = row_rms_norm ~cols:256 [@@gpu.threads 256]
