(* Frontend memory + compute hierarchy smoke: shared, barrier, mad. *)
open Gpu_dsl

let hierarchy_smoke :
  float gpu_array @ aliased read ->
  float gpu_array @ aliased read_write ->
  unit =
fun src dst ->
  let sm = Gpu.shared ~rows:4 ~cols:8 () in
  let tid = Gpu.thread_idx_x () in
  let v = Gpu.load src tid in
  Gpu.shared_store sm tid v;
  Gpu.barrier_cta ();
  let w = Gpu.shared_load sm tid in
  let acc = Gpu.mad_f32 0. w v in
  Gpu.store dst tid acc
