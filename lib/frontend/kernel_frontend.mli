(** Lower versioned [Kernel_ast] to verified semantic IR. *)

val lower : Kernel_ast.t -> Ir.kernel
