(** Semantic IR verification: types, SSA, modes, memory, layouts, uniformity. *)

type error = Verification_error.error = { code : string; message : string }

exception Invalid_kernel of error list

val may_alias :
  ?unique_roots:int list -> Ir.kernel -> Ir.value -> Ir.value -> bool

val verify_kernel : Ir.kernel -> (unit, error list) result
val verify_exn : Ir.kernel -> unit

val check_call :
  Ir.kernel -> Ir.actual list -> (unit, Verification_error.error list) result
