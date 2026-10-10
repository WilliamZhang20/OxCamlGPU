(* Functional constructs that already reach PTX. The adapter is a partial
   evaluator: a function is a compile-time value carrying its captured
   environment, and every call is inlined, so higher-order use and
   polymorphism cost nothing at run time and need no closure representation.
   This fixture pins that coverage; docs/functions.md explains the boundary. *)
open Gpu_dsl

let functional :
  float gpu_array @ aliased read_write ->
  int ->
  unit =
fun y n ->
  let i = Gpu.thread_idx_x () in
  (* A local function applied to a lambda, and applied twice. *)
  let twice f x = f (f x) in
  (* A closure over the kernel formal [n]. *)
  let scale v = v *: n in
  (* Polymorphic, used at both int and float. *)
  let id x = x in
  let idx = twice scale i in
  let piped = id (idx |> twice (fun v -> v +: 1)) in
  Gpu.store y piped (id 1.)
