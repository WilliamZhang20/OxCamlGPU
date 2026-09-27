type 'a gpu_array

module Gpu : sig
  val thread_idx_x : unit -> int
  val load : float gpu_array @ aliased read -> int -> float
  val store : float gpu_array @ aliased read_write -> int -> float -> unit
end
