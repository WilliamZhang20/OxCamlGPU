(** Alias analysis over buffer provenance. *)

val may_alias :
  ?unique_roots:int list -> Ir.kernel -> Ir.value -> Ir.value -> bool
