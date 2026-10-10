(* Index operators must resolve to i32 arithmetic with no range proof. *)
open Gpu_dsl

let idx_ops : float gpu_array @ aliased read_write -> int -> unit =
 fun y n ->
  let i = Gpu.thread_idx_x () in
  Gpu.store y (i *: n +: i %: 4 -: i /: 2) 1.
