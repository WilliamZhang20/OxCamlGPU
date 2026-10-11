(* The GPU surface is a whitelist matched by compiler identity, so an ordinary
   Stdlib function has no GPU meaning however well it typechecks. This is the
   guarantee that writing CPU code in a kernel is a compile error with a
   source location, not a miscompile. *)
open Gpu_dsl

let reject_cpu_primitive :
  float gpu_array @ aliased read_write -> unit =
fun y ->
  print_endline "there is no stdout on a GPU";
  Gpu.store y (Gpu.thread_idx_x ()) 1.
