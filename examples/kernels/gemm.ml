(* Hopper TF32 GEMM. Mid-90s percent of cuBLAS at 4096 cubed, parity at 8192
   cubed. C := A * B, TF32 inputs, f32 accumulate.

   Reading this file
   -----------------
   [hopper_gemm] is the schedule, written once and parameterized by the tile.
   The bindings at the bottom are the only entry points: each applies the
   schedule to one set of tile constants and carries the CTA width as
   [[@@gpu.threads N]]. They exist because shared extents, the WGMMA [n] and
   accumulator indexes must be compile-time constants, so a tile cannot be
   chosen at run time. [gemm_fast] is the tuned default; the rest are named
   after their tile and stage count and exist for the autotune sweep.
   test/gemm_catalog.ml maps a problem size to one of these names.

   [schedule]'s parameters are the kernel's arguments, and their types carry
   OxCaml modes that the compiler checks and turns into GPU facts:

     a_tmap, b_tmap : tensor_map @ aliased read
       A TMA descriptor, the CUtensorMap the host builds to describe a global
       tensor and the tile shape to copy out of it. [cp.async.bulk.tensor]
       reads it. [aliased] says other references to it may exist, [read] says
       this kernel only reads it.
     c : float gpu_array @ unique read_write
       The output buffer. [unique] says no other live reference aliases it,
       which is what lets the backend treat its stores as non-aliasing;
       [read_write] permits the epilogue's stores.
     m_dim, n_dim, k : int
       Problem dimensions.

   Two things here are load-bearing for performance and easy to undo by
   accident: keeping one WGMMA group in flight, and writing C's row stride as
   (N / BN) * BN so the compiler can prove the epilogue's stores are aligned.
   Both are explained where they happen. A third, the grouped CTA order, is
   now a compiler call. docs/gemm-frontend.md collects them.

   Barriers are one indexed set per direction, so the stage count is a free
   parameter rather than a fixed ladder; that is what makes stages=4
   reachable. The shared window exceeds sm_90's 48 KiB static cap, so a launch
   must request it as dynamic shared memory. emit_gemm_ptx --info reports both
   the CTA width and that byte count. *)
open Gpu_dsl

(* [bm]/[bn]/[bk] are the CTA tile, [stages_n] the pipeline depth, [group_m]
   the CTA-order group height. Consumers are one WGMMA warpgroup per 64 rows;
   the remaining warp issues TMA, so a binding's [[@@gpu.threads]] is
   [consumers + 32]. *)
let hopper_gemm ~bm ~bn ~bk ~stages_n ~group_m =
  let consumers = 128 * (bm / 64) in
  let schedule
      (a_tmap : tensor_map @ aliased read)
      (b_tmap : tensor_map @ aliased read)
      (c : float gpu_array @ unique read_write)
      (m_dim : int) (n_dim : int) (k : int) =
    let a_sm = Gpu.shared_wgmma ~rows:(stages_n * bm) ~cols:bk () in
    let b_sm = Gpu.shared_wgmma ~rows:(stages_n * bn) ~cols:bk () in
    (* One barrier pair per stage, indexed by the stage. [full] says the stage
       holds a tile; [empty] says the producers may overwrite it. *)
    let full = Gpu.mbarrier_set stages_n in
    let empty = Gpu.mbarrier_set stages_n in
    let tid = Gpu.thread_idx_x () in
    (* C's row stride. The launch already requires N to be a multiple of BN,
       since the grid is N/BN wide. Writing the stride as (N / BN) * BN states
       that precondition in a form the compiler can use: it proves the stride
       is a multiple of BN, which is what makes the epilogue's column pairs
       provably 8-byte aligned so their stores can vectorize. *)
    let n = n_dim /: bn *: bn in
    let tx_bytes = (bm * bk + bn * bk) * 4 in
    (* Element index of stage s is s * tile_rows * bk. *)
    let a_stride = bm * bk and b_stride = bn * bk in
    (* Grouped CTA order, so one A row-block stays resident across [group_m]
       tile-rows rather than every tile-row re-reading all of B. The group
       height is the tuning choice here; the compiler owns the mapping. *)
    let tiles_m = m_dim /: bm and tiles_n = n /: bn in
    let row_base = Gpu.grouped_tile_m ~group:group_m ~tiles_m ~tiles_n *: bm in
    let col_base = Gpu.grouped_tile_n ~group:group_m ~tiles_m ~tiles_n *: bn in
    (* Warp roles. The split point is this schedule's choice; the warpgroup
       width behind [warpgroup_index] is the hardware's. *)
    let is_consumer = tid < consumers in
    let is_producer = tid >= consumers in
    let is_elect = tid = consumers in
    let wg = Gpu.warpgroup_index () in
    let acc = Gpu.wgmma_acc ~n:bn () in
    for s = 0 to stages_n - 1 do
      Gpu.mbarrier_init_elect (Gpu.mbarrier_slot full s) 1 is_elect;
      Gpu.mbarrier_init_elect (Gpu.mbarrier_slot empty s) consumers is_elect
    done;
    let wait_full stage parity =
      Gpu.mbarrier_try_wait_parity (Gpu.mbarrier_slot full stage) parity;
      Gpu.fence_proxy_async ()
    in
    let release stage = Gpu.mbarrier_arrive (Gpu.mbarrier_slot empty stage) in
    let prefetch stage parity k_col row col =
      let full_s = Gpu.mbarrier_slot full stage in
      Gpu.mbarrier_try_wait_parity (Gpu.mbarrier_slot empty stage) parity;
      Gpu.mbarrier_arrive_expect_tx_elect full_s tx_bytes is_elect;
      Gpu.tma_load_2d_elect a_sm (stage *: a_stride) a_tmap k_col row full_s is_elect;
      Gpu.tma_load_2d_elect b_sm (stage *: b_stride) b_tmap k_col col full_s is_elect
    in
    Gpu.barrier_cta ();
    if is_consumer then
      for s = 0 to stages_n - 1 do
        Gpu.mbarrier_arrive (Gpu.mbarrier_slot empty s)
      done;
    Gpu.barrier_cta ();
    if is_producer then prefetch 0 0 0 row_base col_base;
    let tiles = k /: bk in
    for t = 0 to tiles -: 1 do
      let stage = t %: stages_n in
      let parity = t /: stages_n %: 2 in
      let next_t = t +: 1 in
      (* (t - 1) mod stages, biased so t = 0 stays in range; [not_first]
         guards the only use. *)
      let prev_stage = (t +: stages_n -: 1) %: stages_n in
      let not_first = t > 0 in
      if is_consumer then begin
        wait_full stage parity;
        let a_base = (stage *: bm +: wg *: 64) *: bk in
        let b_base = stage *: bn *: bk in
        Gpu.wgmma_fence ();
        (* Accumulators start at 0, so scale accumulates on every K-slice. *)
        for ki = 0 to bk / 8 - 1 do
          let off = 8 * ki in
          Gpu.wgmma_mma_tf32 acc
            (Gpu.gmma_descriptor a_sm (a_base +: off))
            (Gpu.gmma_descriptor b_sm (b_base +: off))
            ~n:bn ~scale:true
        done;
        Gpu.wgmma_commit_group ()
      end;
      if is_producer && next_t < tiles then
        prefetch (next_t %: stages_n) (next_t /: stages_n %: 2)
          (t *: bk +: bk) row_base col_base;
      if is_consumer then begin
        (* Keep one committed group in flight, so this tile's WGMMA overlaps
           the next tile's barrier wait instead of draining into it. A stage
           is only safe to hand back once the group that read it has retired,
           which [wait_group 1] establishes for the previous tile. *)
        Gpu.wgmma_wait_group 1;
        if not_first then release prev_stage
      end
    done;
    (* Drain the last group and hand back the stage it read. *)
    if is_consumer then begin
      Gpu.wgmma_wait_group 0;
      release ((tiles -: 1) %: stages_n)
    end;
    if is_consumer then begin
      (* This warpgroup owns 64 rows of the CTA tile; where each accumulator
         register sits inside them is the WGMMA layout, which the compiler
         supplies. Registers 4g and 4g+1 are column-adjacent, so the backend
         fuses each pair into one 8-byte store. *)
      let row0 = row_base +: wg *: 64 in
      for g = 0 to bn / 8 - 1 do
        for j = 0 to 3 do
          let i = 4 * g + j in
          let row = row0 +: Gpu.wgmma_acc_row acc i in
          let col = col_base +: Gpu.wgmma_acc_col acc i in
          Gpu.store c (row *: n +: col) (Gpu.wgmma_acc_get acc i)
        done
      done
    end
  in
  schedule

let gemm_fast =
  hopper_gemm ~bm:128 ~bn:256 ~bk:32 ~stages_n:4 ~group_m:16
[@@gpu.threads 288]

let gemm_bm128_bn256_s3 =
  hopper_gemm ~bm:128 ~bn:256 ~bk:32 ~stages_n:3 ~group_m:16
[@@gpu.threads 288]

let gemm_bm128_bn256_s2 =
  hopper_gemm ~bm:128 ~bn:256 ~bk:32 ~stages_n:2 ~group_m:16
[@@gpu.threads 288]

let gemm_bm128_bn128_s3 =
  hopper_gemm ~bm:128 ~bn:128 ~bk:32 ~stages_n:3 ~group_m:16
[@@gpu.threads 288]

let gemm_bm128_bn128_s2 =
  hopper_gemm ~bm:128 ~bn:128 ~bk:32 ~stages_n:2 ~group_m:16
[@@gpu.threads 288]

let gemm_bm64_bn256_s3 =
  hopper_gemm ~bm:64 ~bn:256 ~bk:32 ~stages_n:3 ~group_m:16
[@@gpu.threads 160]

let gemm_bm64_bn256_s2 =
  hopper_gemm ~bm:64 ~bn:256 ~bk:32 ~stages_n:2 ~group_m:16
[@@gpu.threads 160]

let gemm_bm256_bn128_s3 =
  hopper_gemm ~bm:256 ~bn:128 ~bk:32 ~stages_n:3 ~group_m:16
[@@gpu.threads 544]

let gemm_bm256_bn128_s2 =
  hopper_gemm ~bm:256 ~bn:128 ~bk:32 ~stages_n:2 ~group_m:16
[@@gpu.threads 544]
