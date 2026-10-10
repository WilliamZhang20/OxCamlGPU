(** Host-call argument checking against kernel formals. *)

val check_call :
  Ir.kernel -> Ir.actual list -> (unit, Verification_error.error list) result
