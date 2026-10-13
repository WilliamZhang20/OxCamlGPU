(* CTA width is launch metadata, not an operation. Put it on the kernel binding:
   [let kernel ... = ... [@@gpu.threads 288]]. The adapter stores it as the
   [.gpu] [threads] field and the PTX backend emits [.reqntid]. *)

type 'a gpu_array
type 'a gpu_shared
type gpu_mbarrier
type wgmma_acc

(** Element types. [float] is the f32 tag; [bf16] and [f16] have no OCaml
    scalar because no register ever holds one — they are storage formats that
    TMA moves and WGMMA reads straight out of shared memory, which is also how
    a CUTLASS bf16 GEMM works. The accumulator stays f32 at every width.

    Width is therefore never written down in a kernel. It rides on the handle:
    a [bf16 tensor_map] formal makes the tile it feeds a [bf16 gpu_shared],
    which makes the descriptor a [bf16 gmma_desc], which makes [wgmma_mma]
    emit [m64nNk16.f32.bf16.bf16]. One name per operation, and the element
    type picks the instruction. *)
type bf16
type f16

(** A CUtensorMap for a tensor of ['a]. The element type is part of the map
    the host built, so typing it here is what grounds the width for the rest
    of the kernel and stops an f32 map from feeding a bf16 WGMMA. *)
type 'a tensor_map

(** A packed GMMA shared-operand descriptor for a tile of ['a]. Typed rather
    than [int64] so A and B cannot disagree and so [wgmma_mma] knows which
    instruction to emit. *)
type 'a gmma_desc

(** Index arithmetic, 32-bit and wrapping, at the top level so an
    [open Gpu_dsl] makes it infix. Plain [+] and [*] stay range-checked: the
    compiler proves the result fits i32 and refuses when it cannot, which is
    right for [i + 1] on a thread index. It cannot bound a problem dimension
    times a row, so index math that genuinely is 32-bit asks with these
    instead of being forced through [Int32] and its conversions. *)
val ( +: ) : int -> int -> int
val ( -: ) : int -> int -> int
val ( *: ) : int -> int -> int
val ( /: ) : int -> int -> int
val ( %: ) : int -> int -> int

module Gpu : sig
  val thread_idx_x : unit -> int
  val global_idx_x : unit -> int
  val block_idx_x : unit -> int
  val block_idx_y : unit -> int
  (** Which warpgroup the calling thread is in. 128 is the warpgroup width. *)
  val warpgroup_index : unit -> int
  (** This CTA's tile row and column under a grouped CTA order: [group]
      tile-rows are walked before advancing along N, so one A row-block stays
      resident across a group instead of every tile-row re-reading all of B.
      Covers every tile exactly once, including a short last group. The group
      height is a tuning choice; the mapping is not. *)
  val grouped_tile_m : group:int -> tiles_m:int -> tiles_n:int -> int
  val grouped_tile_n : group:int -> tiles_m:int -> tiles_n:int -> int

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
  (** K-major + 128B swizzle shared (TMA/WGMMA). The element type comes from
      use — normally from the ['a tensor_map] that fills it. A swizzle period
      is 128 bytes at every width, so [cols] is 32 for f32 and 64 for bf16. *)
  val shared_wgmma : rows:int -> cols:int -> unit -> 'a gpu_shared
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

  (** Approximate float intrinsics: the hardware special-function unit, one
      instruction each. Choosing approximation is a numerical decision, so it
      is in the name rather than inferred — [sqrt], [/.] and [-.] from the
      Stdlib lower to the correctly-rounded [sqrt.rn], [div.rn] and [sub.rn],
      and [Float.max]/[Float.min] keep their OCaml NaN behaviour.

      These are base-2 because that is what the hardware has. Natural
      exponentials compose: [exp x] is [exp2_approx (x *. 1.4426950408889634)],
      which keeps the change of base visible at the call site. *)
  val rsqrt_approx : float -> float
  val exp2_approx : float -> float
  val log2_approx : float -> float
  val tanh_approx : float -> float
  val recip_approx : float -> float

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
    'a gpu_shared -> int -> 'a tensor_map @ aliased read ->
    int -> int -> gpu_mbarrier -> unit
  val tma_load_2d_elect :
    'a gpu_shared -> int -> 'a tensor_map @ aliased read ->
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
  (** Shared-memory operand descriptor for [wgmma_mma], at an element offset
      into a [shared_wgmma] tile. The packed leading/stride/layout fields
      follow from that tile's extents and swizzle, so the compiler derives
      them rather than asking. Every field is a byte count, which is why the
      same derivation serves every element width. *)
  val gmma_descriptor : 'a gpu_shared -> int -> 'a gmma_desc

  (** [acc] += A @ B, f32 accumulate. The operands' element type selects the
      instruction: [float] is tf32 with K=8, [bf16] and [f16] are K=16. The
      m64, the accumulator width and the register layout are the same across
      all three, so [wgmma_acc_row]/[wgmma_acc_col] and the epilogue do not
      change with the input width. *)
  val wgmma_mma :
    wgmma_acc -> 'a gmma_desc -> 'a gmma_desc -> n:int -> scale:bool -> unit
  val wgmma_commit_group : unit -> unit
  val wgmma_wait_group : int -> unit
end
