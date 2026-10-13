(* A software store into a 128-byte-swizzled tile. The adapter accepts this —
   nothing about it is ill-typed — and the backend would address the tile
   linearly while GMMA reads it permuted, so the WGMMA below would consume the
   wrong bytes and return a plausible wrong answer. The verifier is what has
   to catch it, and this fixture is what keeps it caught.

   Lifting the rule means teaching PTX emission to apply the swizzle in
   software, at which point the four-value swizzle enum becomes a real limit;
   docs/roadmap.md has the argument. *)
open Gpu_dsl

let swizzled_shared :
  float gpu_array @ aliased read ->
  float gpu_array @ unique read_write ->
  unit =
fun x c ->
  let sm = Gpu.shared_wgmma ~rows:64 ~cols:32 () in
  let tid = Gpu.thread_idx_x () in
  Gpu.shared_store sm tid (Gpu.load x tid);
  Gpu.barrier_cta ();
  let acc = Gpu.wgmma_acc ~n:8 () in
  Gpu.wgmma_fence ();
  Gpu.wgmma_mma acc
    (Gpu.gmma_descriptor sm 0)
    (Gpu.gmma_descriptor sm 0)
    ~n:8 ~scale:false;
  Gpu.wgmma_commit_group ();
  Gpu.wgmma_wait_group 0;
  Gpu.store c tid (Gpu.wgmma_acc_get acc 0)
[@@gpu.threads 128]
