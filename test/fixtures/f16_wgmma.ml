(* f16 is the third width the WGMMA path accepts, and nothing about it differs
   from bf16 except the instruction suffix. This fixture pins that: one
   m64n8k16 f16 WGMMA, small enough to read, so the claim is a compiled
   artifact rather than a sentence in a table.

   It is a fixture and not an example because it computes a partial product
   over one K-slice; there is no hardware test for f16, only this and ptxas.
   examples/kernels/gemm_bf16.ml is the width that is verified on an H100. *)
open Gpu_dsl

let f16_wgmma :
  f16 tensor_map @ aliased read ->
  f16 tensor_map @ aliased read ->
  float gpu_array @ unique read_write ->
  unit =
fun a_map b_map c ->
  (* 128-byte rows: 64 f16 columns, the same period as 32 f32 columns. *)
  let a_sm = Gpu.shared_wgmma ~rows:64 ~cols:64 () in
  let b_sm = Gpu.shared_wgmma ~rows:8 ~cols:64 () in
  let full = Gpu.mbarrier () in
  let tid = Gpu.thread_idx_x () in
  let elect = tid = 0 in
  Gpu.mbarrier_init_elect full 1 elect;
  Gpu.barrier_cta ();
  Gpu.mbarrier_arrive_expect_tx_elect full ((64 * 64 + 8 * 64) * 2) elect;
  Gpu.tma_load_2d_elect a_sm 0 a_map 0 0 full elect;
  Gpu.tma_load_2d_elect b_sm 0 b_map 0 0 full elect;
  Gpu.mbarrier_try_wait_parity full 0;
  Gpu.fence_proxy_async ();
  let acc = Gpu.wgmma_acc ~n:8 () in
  Gpu.wgmma_fence ();
  Gpu.wgmma_mma acc
    (Gpu.gmma_descriptor a_sm 0)
    (Gpu.gmma_descriptor b_sm 0)
    ~n:8 ~scale:false;
  Gpu.wgmma_commit_group ();
  Gpu.wgmma_wait_group 0;
  for i = 0 to 3 do
    Gpu.store c
      (Gpu.wgmma_acc_row acc i *: 8 +: Gpu.wgmma_acc_col acc i)
      (Gpu.wgmma_acc_get acc i)
  done
[@@gpu.threads 128]
