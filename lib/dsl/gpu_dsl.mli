(* CTA width is launch metadata, not an operation. Put it on the kernel binding:
   [let kernel ... = ... [@@gpu.threads 288]]. The adapter stores it as the
   [.gpu] [threads] field and the PTX backend emits [.reqntid]. *)

type 'a gpu_array
type 'a gpu_shared
type gpu_mbarrier
type tensor_map
type wgmma_acc

module Gpu : sig
  val thread_idx_x : unit -> int
  val global_idx_x : unit -> int
  val block_idx_x : unit -> int
  val block_idx_y : unit -> int

  val load : float gpu_array @ aliased read -> int -> float
  val store : float gpu_array @ aliased read_write -> int -> float -> unit
  val load_masked : float gpu_array @ aliased read -> int -> int -> float
  val store_masked : float gpu_array @ aliased read_write -> int -> int -> float -> unit

  val load_f32x4 :
    float gpu_array @ aliased read -> int -> float * float * float * float
  val store_f32x4 :
    float gpu_array @ aliased read_write ->
    int -> float -> float -> float -> float -> unit

  (** CTA shared allocation. [rows]/[cols] must be compile-time positive constants. *)
  val shared : rows:int -> cols:int -> unit -> float gpu_shared
  (** K-major + 128B swizzle shared (TMA/WGMMA). *)
  val shared_wgmma : rows:int -> cols:int -> unit -> float gpu_shared
  val shared_load : float gpu_shared -> int -> float
  val shared_load_f32x4 :
    float gpu_shared -> int -> float * float * float * float
  val shared_store : float gpu_shared -> int -> float -> unit
  val shared_store_f32x4 :
    float gpu_shared -> int -> float -> float -> float -> float -> unit

  val barrier_cta : unit -> unit
  val cp_async_f32x4 :
    float gpu_shared -> int -> float gpu_array @ aliased read -> int -> unit
  val cp_async_commit : unit -> unit
  val cp_async_wait : unit -> unit

  (** [mad_f32 acc a b] = [acc +. a *. b]. Explicit FMA; not inferred. *)
  val mad_f32 : float -> float -> float -> float

  val warp_sum_f32 : float -> float
  val store_grid_leader : float gpu_array @ unique write -> float -> unit

  (* --- Hopper memory / compute hierarchy (author-scheduled) --- *)
  val mbarrier : unit -> gpu_mbarrier
  (** [mbarrier_set n] allocates [n] mbarriers; [mbarrier_slot set i] is the
      one at index [i], which may be computed at run time. A pipelined
      schedule indexes its stage barrier directly instead of branching over a
      fixed ladder of separately named barriers. [i] must be in range. *)
  val mbarrier_set : int -> gpu_mbarrier
  val mbarrier_slot : gpu_mbarrier -> int -> gpu_mbarrier
  val mbarrier_init : gpu_mbarrier -> int -> unit
  val mbarrier_init_elect : gpu_mbarrier -> int -> bool -> unit
  val mbarrier_arrive_expect_tx : gpu_mbarrier -> int -> unit
  val mbarrier_arrive_expect_tx_elect : gpu_mbarrier -> int -> bool -> unit
  val mbarrier_arrive : gpu_mbarrier -> unit
  val mbarrier_arrive_elect : gpu_mbarrier -> bool -> unit
  val mbarrier_try_wait_parity : gpu_mbarrier -> int -> unit

  val tma_load_2d :
    float gpu_shared -> int -> tensor_map @ aliased read ->
    int -> int -> gpu_mbarrier -> unit
  val tma_load_2d_elect :
    float gpu_shared -> int -> tensor_map @ aliased read ->
    int -> int -> gpu_mbarrier -> bool -> unit
  val fence_proxy_async : unit -> unit

  val wgmma_fence : unit -> unit
  (** Allocate BN/2 TF32 WGMMA accumulator registers (n = BN in {8,...,256}). *)
  val wgmma_acc : n:int -> unit -> wgmma_acc
  (** Read accumulator register [i] (0 .. n/2 - 1). [i] must be a constant. *)
  val wgmma_acc_get : wgmma_acc -> int -> float
  (** Row and column of accumulator register [i] inside this warpgroup's
      m64 x n tile, for the calling lane. With [wgmma_acc_get] these let an
      epilogue walk the accumulator without hand-encoding the WGMMA register
      layout. Which 64 rows a warpgroup owns is still the author's choice. *)
  val wgmma_acc_row : wgmma_acc -> int -> int
  val wgmma_acc_col : wgmma_acc -> int -> int
  (** Shared-memory operand descriptor for [wgmma_mma_tf32], at an element
      offset into a [shared_wgmma] tile. The packed leading/stride/layout
      fields follow from that tile's extents and swizzle, so the compiler
      derives them rather than asking. *)
  val gmma_descriptor : float gpu_shared -> int -> int64
  val wgmma_mma_tf32 :
    wgmma_acc -> int64 -> int64 -> n:int -> scale:bool -> unit
  val wgmma_commit_group : unit -> unit
  val wgmma_wait_group : int -> unit
end
