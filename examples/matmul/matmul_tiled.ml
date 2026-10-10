(* Hopper TF32 CTA GEMM. Mid-90s percent of cuBLAS at 4096 cubed, parity at
   8192 cubed.

   One schedule, specialized at compile time. Shared extents, WGMMA [n], and
   accumulator indexes have to be constants, so each catalog tile is a binding
   that applies [hopper_gemm] to constants.

   [stages_n] barriers are allocated as one indexed set per direction, so the
   stage count is a free parameter rather than a fixed ladder of separately
   named barriers. That is what makes stages=4 reachable, and it is what
   [choose_config] picks; see matmul_config.ml for the measurements.

   Three things here are load-bearing for performance, each explained where it
   happens: the grouped CTA order, keeping one WGMMA group in flight, and
   writing C's row stride so the compiler can prove the epilogue's stores are
   aligned. docs/matmul-frontend.md collects them.

   The shared window is 148112 bytes, over sm_90's 48 KiB static cap, so the
   launch must request it as dynamic shared memory; emit_matmul_ptx --info
   reports the number. *)
open Gpu_dsl

let hopper_gemm ~bm ~bn ~bk ~stages_n ~group_m ~producers:_ =
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
    let tid = Int32.of_int (Gpu.thread_idx_x ()) in
    let bx = Int32.of_int (Gpu.block_idx_x ()) in
    let by = Int32.of_int (Gpu.block_idx_y ()) in
    let k = Int32.of_int k in
    let bm32 = Int32.of_int bm and bn32 = Int32.of_int bn and bk32 = Int32.of_int bk in
    (* C's row stride. The launch already requires N to be a multiple of BN,
       since the grid is N/BN wide. Writing the stride as (N / BN) * BN states
       that precondition in a form the compiler can use: it proves the stride
       is a multiple of BN, which is what makes the epilogue's column pairs
       provably 8-byte aligned so their stores can vectorize. *)
    let n = Int32.mul (Int32.div (Int32.of_int n_dim) bn32) bn32 in
    let stages = Int32.of_int stages_n in
    let n_cons = Int32.of_int consumers in
    let tx_bytes = (bm * bk + bn * bk) * 4 in
    (* Element index of stage s is s * tile_rows * bk. *)
    let a_stride = Int32.of_int (bm * bk) in
    let b_stride = Int32.of_int (bn * bk) in
    (* Grouped CTA order. In the default row-major order every tile-row
       streams the whole of B, so B's DRAM traffic scales with the number of
       tile-rows and gets worse as the problem grows. Walking [group_m]
       tile-rows before advancing along N lets one A row-block stay resident
       across a group and cuts B's re-reads by the same factor. The mapping is
       the standard grouped bijection, so every tile is still covered once. *)
    let tiles_n = Int32.div n bn32 in
    let tiles_m = Int32.div (Int32.of_int m_dim) bm32 in
    let group = Int32.of_int group_m in
    let pid = Int32.add (Int32.mul by tiles_n) bx in
    let per_group = Int32.mul group tiles_n in
    let group_first = Int32.mul (Int32.div pid per_group) group in
    let group_left = Int32.sub tiles_m group_first in
    (* The last group is short when tiles_m is not a multiple of the group. *)
    let group_rows = if group_left < group then group_left else group in
    let tile_m = Int32.add group_first (Int32.rem pid group_rows) in
    let tile_n = Int32.div (Int32.rem pid per_group) group_rows in
    let row_base = Int32.mul tile_m bm32 in
    let col_base = Int32.mul tile_n bn32 in
    let is_consumer = tid < n_cons in
    let is_producer = tid >= n_cons in
    let is_elect = tid = n_cons in
    let wg = Int32.div tid 128l in
    let lane_wg = Int32.rem tid 128l in
    let acc = Gpu.wgmma_acc ~n:bn () in
    for s = 0 to stages_n - 1 do
      Gpu.mbarrier_init_elect (Gpu.mbarrier_slot full s) 1 is_elect;
      Gpu.mbarrier_init_elect (Gpu.mbarrier_slot empty s) consumers is_elect
    done;
    let wait_full stage parity =
      Gpu.mbarrier_try_wait_parity
        (Gpu.mbarrier_slot full (Int32.to_int stage)) (Int32.to_int parity);
      Gpu.fence_proxy_async ()
    in
    let release stage =
      Gpu.mbarrier_arrive (Gpu.mbarrier_slot empty (Int32.to_int stage))
    in
    let prefetch stage parity k_col row col =
      let a_idx = Int32.to_int (Int32.mul stage a_stride) in
      let b_idx = Int32.to_int (Int32.mul stage b_stride) in
      let k_col = Int32.to_int k_col in
      let row = Int32.to_int row in
      let col = Int32.to_int col in
      let s = Int32.to_int stage in
      let full_s = Gpu.mbarrier_slot full s in
      Gpu.mbarrier_try_wait_parity
        (Gpu.mbarrier_slot empty s) (Int32.to_int parity);
      Gpu.mbarrier_arrive_expect_tx_elect full_s tx_bytes is_elect;
      Gpu.tma_load_2d_elect a_sm a_idx a_tmap k_col row full_s is_elect;
      Gpu.tma_load_2d_elect b_sm b_idx b_tmap k_col col full_s is_elect
    in
    Gpu.barrier_cta ();
    if is_consumer then
      for s = 0 to stages_n - 1 do
        Gpu.mbarrier_arrive (Gpu.mbarrier_slot empty s)
      done;
    Gpu.barrier_cta ();
    if is_producer then prefetch 0l 0l 0l row_base col_base;
    let tiles = Int32.div k bk32 in
    let tiles_hi = Int32.to_int (Int32.sub tiles 1l) in
    let last_stage = Int32.rem (Int32.sub tiles 1l) stages in
    for t = 0 to tiles_hi do
      let t32 = Int32.of_int t in
      let k0 = Int32.mul t32 bk32 in
      let stage = Int32.rem t32 stages in
      let next_t = Int32.add t32 1l in
      let next_stage = Int32.rem next_t stages in
      let parity = Int32.rem (Int32.div t32 stages) 2l in
      let parity_next = Int32.rem (Int32.div next_t stages) 2l in
      let more = next_t < tiles in
      (* (t - 1) mod stages, biased so t = 0 stays in range; [not_first]
         guards the only use. *)
      let prev_stage = Int32.rem (Int32.add t32 (Int32.sub stages 1l)) stages in
      let not_first = t32 > 0l in
      if is_consumer then begin
        wait_full stage parity;
        let a_row = Int32.add (Int32.mul stage bm32) (Int32.mul wg 64l) in
        let a_base = Int32.mul a_row bk32 in
        let b_base = Int32.mul (Int32.mul stage bn32) bk32 in
        Gpu.wgmma_fence ();
        (* Accumulators start at 0, so scale accumulates on every K-slice. *)
        for ki = 0 to bk / 8 - 1 do
          let off = Int32.of_int (8 * ki) in
          let desc_a =
            Gpu.gmma_descriptor a_sm (Int32.to_int (Int32.add a_base off))
          in
          let desc_b =
            Gpu.gmma_descriptor b_sm (Int32.to_int (Int32.add b_base off))
          in
          Gpu.wgmma_mma_tf32 acc desc_a desc_b ~n:bn ~scale:true
        done;
        Gpu.wgmma_commit_group ()
      end;
      if is_producer && more then
        prefetch next_stage parity_next (Int32.add k0 bk32) row_base col_base;
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
      release last_stage
    end;
    if is_consumer then begin
      let wid = Int32.div lane_wg 32l in
      let lane = Int32.rem lane_wg 32l in
      let frag_row = Int32.add (Int32.mul wid 16l) (Int32.div lane 4l) in
      let frag_col = Int32.mul (Int32.rem lane 4l) 2l in
      let row0 = Int32.add row_base (Int32.mul wg 64l) in
      let row_a = Int32.add row0 frag_row in
      let row_b = Int32.add row_a 8l in
      let store row col value =
        Gpu.store c (Int32.to_int (Int32.add (Int32.mul row n) col)) value
      in
      (* BN/8 fragment groups, four accumulator registers each. Each group's
         two column-adjacent stores are consecutive f32, so the backend fuses
         them into one 8-byte vector store. *)
      for g = 0 to bn / 8 - 1 do
        let col0 = Int32.add col_base (Int32.add frag_col (Int32.of_int (8 * g))) in
        let col1 = Int32.add col0 1l in
        let base = 4 * g in
        store row_a col0 (Gpu.wgmma_acc_get acc base);
        store row_a col1 (Gpu.wgmma_acc_get acc (base + 1));
        store row_b col0 (Gpu.wgmma_acc_get acc (base + 2));
        store row_b col1 (Gpu.wgmma_acc_get acc (base + 3))
      done
    end
  in
  schedule

let matmul_tiled =
  hopper_gemm ~bm:128 ~bn:256 ~bk:32 ~stages_n:4 ~group_m:16 ~producers:32
[@@gpu.threads 288]

let matmul_bm128_bn256_s3 =
  hopper_gemm ~bm:128 ~bn:256 ~bk:32 ~stages_n:3 ~group_m:16 ~producers:32
[@@gpu.threads 288]

let matmul_bm128_bn256_s2 =
  hopper_gemm ~bm:128 ~bn:256 ~bk:32 ~stages_n:2 ~group_m:16 ~producers:32
[@@gpu.threads 288]

let matmul_bm128_bn128_s3 =
  hopper_gemm ~bm:128 ~bn:128 ~bk:32 ~stages_n:3 ~group_m:16 ~producers:32
[@@gpu.threads 288]

let matmul_bm128_bn128_s2 =
  hopper_gemm ~bm:128 ~bn:128 ~bk:32 ~stages_n:2 ~group_m:16 ~producers:32
[@@gpu.threads 288]

let matmul_bm64_bn256_s3 =
  hopper_gemm ~bm:64 ~bn:256 ~bk:32 ~stages_n:3 ~group_m:16 ~producers:32
[@@gpu.threads 160]

let matmul_bm64_bn256_s2 =
  hopper_gemm ~bm:64 ~bn:256 ~bk:32 ~stages_n:2 ~group_m:16 ~producers:32
[@@gpu.threads 160]

let matmul_bm256_bn128_s3 =
  hopper_gemm ~bm:256 ~bn:128 ~bk:32 ~stages_n:3 ~group_m:16 ~producers:32
[@@gpu.threads 544]

let matmul_bm256_bn128_s2 =
  hopper_gemm ~bm:256 ~bn:128 ~bk:32 ~stages_n:2 ~group_m:16 ~producers:32
[@@gpu.threads 544]
