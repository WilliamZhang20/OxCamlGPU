(** PTX text emission. *)

(** Emit PTX from a physicalized target kernel. *)
val emit_physical : Physical_ir.kernel -> string

(** Emit PTX from a target kernel (runs physicalization). *)
val emit_target : Ptx_ir.kernel -> string

(** Full pipeline from semantic IR: lower -> physicalize -> emit. *)
val emit : Ir.kernel -> string

(** Shared memory a kernel's allocations occupy. All allocations share one
    window; [dynamic_bytes] is what a launch must request as dynamic shared
    memory, and is 0 when the window fits the 48 KiB static limit. *)
type shared_requirement = { total_bytes : int; dynamic_bytes : int }

val shared_requirement : Physical_ir.kernel -> shared_requirement
