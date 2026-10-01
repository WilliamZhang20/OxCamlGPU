type 'a gpu_array

module Gpu : sig
  val thread_idx_x : unit -> int
  val global_idx_x : unit -> int
  val load : float gpu_array @ aliased read -> int -> float
  val store : float gpu_array @ aliased read_write -> int -> float -> unit
  val load_masked : float gpu_array @ aliased read -> int -> int -> float
  val store_masked : float gpu_array @ aliased read_write -> int -> int -> float -> unit
  val warp_sum_f32 : float -> float
  val store_grid_leader : float gpu_array @ unique write -> float -> unit
end
