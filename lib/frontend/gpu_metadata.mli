(** Versioned [.gpu] metadata encode/decode. *)

exception Parse_error of string

val encode : Kernel_ast.t -> string
val parse : string -> Kernel_ast.t
