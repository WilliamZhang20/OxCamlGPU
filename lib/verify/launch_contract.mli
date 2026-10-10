(** Host-facing launch contract over buffer actuals. *)

type buffer_actual = {
  buffer_id : int;
  ownership : Gpu_mode.ownership;
  permission : Gpu_mode.permission;
  locality : Gpu_mode.locality;
  domain_portability : Gpu_mode.domain_portability;
  gpu_boundary : Gpu_mode.gpu_boundary;
}

val to_actual : buffer_actual -> Ir.actual

val buffer :
  ?ownership:Gpu_mode.ownership ->
  ?permission:Gpu_mode.permission ->
  ?locality:Gpu_mode.locality ->
  ?domain_portability:Gpu_mode.domain_portability ->
  ?gpu_boundary:Gpu_mode.gpu_boundary ->
  int ->
  buffer_actual

val check :
  Ir.kernel ->
  buffer_actual list ->
  (unit, Verification_error.error list) result

val check_exn : Ir.kernel -> buffer_actual list -> unit
