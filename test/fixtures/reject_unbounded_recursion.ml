(* Recursion is inlined, so it only terminates when the controlling test folds
   at compile time. Here the bound is a thread index, so the base case stays a
   runtime branch, both arms are traced, and the inlining never ends. The
   adapter must say so with a source location rather than run out of stack. *)
open Gpu_dsl

let reject_unbounded_recursion :
  float gpu_array @ aliased read_write -> unit =
fun y ->
  let i = Gpu.thread_idx_x () in
  let rec go n acc = if n = 0 then acc else go (n -: 1) (acc +: n) in
  Gpu.store y (go i 0) 1.
