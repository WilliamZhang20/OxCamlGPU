(** Lower semantic IR to target [Ptx_ir], expanding strategies as needed. *)

val lower : Ir.kernel -> Ptx_ir.kernel
