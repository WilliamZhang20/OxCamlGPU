(* Hopper BF16 GEMM. C := A * B, bf16 inputs, f32 accumulate.

   This is gemm.ml's schedule at the other input width, and the point of the
   file is how little changes. Compare them side by side: the pipeline, the
   barrier sets, the warp roles, the grouped CTA order and the whole epilogue
   are identical, because none of them depend on how wide an input element is.
   Four things do, and each is a number rather than a different operation:

     tensor_map       [bf16] instead of [float] on the two formals. That one
                      word is what grounds the width: it makes the tiles
                      [bf16 gpu_shared], the descriptors [bf16 gmma_desc],
                      and [wgmma_mma] emit [m64nNk16.f32.bf16.bf16].
     bk               64, not 32. A 128-byte swizzle period is 64 bf16
                      columns. The shared footprint is unchanged — half the
                      element size, twice the K depth, same bytes.
     wgmma_k          16, not 8. The instruction's K, which is the one thing
                      the element type changes about the math.
     tx_bytes         two bytes per element instead of four.

   Everything else the compiler derives: the GMMA descriptor's leading and
   stride fields are byte counts, so they come out the same at both widths,
   and the f32 accumulator layout behind [wgmma_acc_row]/[wgmma_acc_col] does
   not depend on K at all.

   What is *not* shared is this file's existence. One source schedule cannot
   yet be polymorphic over the element type, because the adapter reads types
   off a single Typedtree and does not substitute type variables while
   inlining, so a ['a tensor_map] formal stays unresolved. That is a frontend
   gap, not a GPU one; docs/roadmap.md tracks it. Read gemm.ml for why the
   schedule is shaped the way it is — the comments there are not repeated. *)
open Gpu_dsl

let hopper_gemm_bf16 ~bm ~bn ~bk ~stages_n ~group_m =
  let consumers = 128 * (bm / 64) in
  (* WGMMA consumes 16 bf16 elements of K per instruction. *)
  let wgmma_k = 16 in
  let schedule
      (a_tmap : bf16 tensor_map @ aliased read)
      (b_tmap : bf16 tensor_map @ aliased read)
      (c : float gpu_array @ unique read_write)
      (m_dim : int) (n_dim : int) (k : int) =
    let a_sm = Gpu.shared_wgmma ~rows:(stages_n * bm) ~cols:bk () in
    let b_sm = Gpu.shared_wgmma ~rows:(stages_n * bn) ~cols:bk () in
    let full = Gpu.mbarrier_set stages_n in
    let empty = Gpu.mbarrier_set stages_n in
    let tid = Gpu.thread_idx_x () in
    (* As in gemm.ml: stating N's precondition this way is what proves the
       epilogue's column pairs are 8-byte aligned. C is f32 at either width. *)
    let n = n_dim /: bn *: bn in
    let tx_bytes = (bm * bk + bn * bk) * 2 in
    let a_stride = bm * bk and b_stride = bn * bk in
    let tiles_m = m_dim /: bm and tiles_n = n /: bn in
    let row_base = Gpu.grouped_tile_m ~group:group_m ~tiles_m ~tiles_n *: bm in
    let col_base = Gpu.grouped_tile_n ~group:group_m ~tiles_m ~tiles_n *: bn in
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
      let prev_stage = (t +: stages_n -: 1) %: stages_n in
      let not_first = t > 0 in
      if is_consumer then begin
        wait_full stage parity;
        let a_base = (stage *: bm +: wg *: 64) *: bk in
        let b_base = stage *: bn *: bk in
        Gpu.wgmma_fence ();
        for ki = 0 to bk / wgmma_k - 1 do
          let off = wgmma_k * ki in
          Gpu.wgmma_mma acc
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
        Gpu.wgmma_wait_group 1;
        if not_first then release prev_stage
      end
    done;
    if is_consumer then begin
      Gpu.wgmma_wait_group 0;
      release ((tiles -: 1) %: stages_n)
    end;
    if is_consumer then begin
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

let gemm_bf16_fast =
  hopper_gemm_bf16 ~bm:128 ~bn:256 ~bk:64 ~stages_n:4 ~group_m:16
[@@gpu.threads 288]
