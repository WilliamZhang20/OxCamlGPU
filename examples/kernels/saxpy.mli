type f32
type i32
type 'a gpu_array

module Gpu : sig
  val thread_idx_x : unit -> i32
  val load : f32 gpu_array @ aliased read -> i32 -> f32
  val store : f32 gpu_array @ aliased read_write -> i32 -> f32 -> unit
end

module F32 : sig
  val ( +. ) : f32 -> f32 -> f32
  val ( *. ) : f32 -> f32 -> f32
end

val saxpy :
  f32 gpu_array @ aliased read ->
  f32 gpu_array @ unique read_write ->
  f32 ->
  unit
