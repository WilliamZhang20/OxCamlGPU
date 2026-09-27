(* Surface declaration authored in OxCaml syntax. *)
type f32
type 'a gpu_array

val saxpy :
  f32 gpu_array @ aliased read ->
  f32 gpu_array @ unique read_write ->
  f32 ->
  unit
