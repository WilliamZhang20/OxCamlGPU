(** Compile a verified GPU IR kernel to PTX. *)

(** PTX text plus the launch facts that are not recoverable from it. *)
type compiled = {
  ptx : string;
  (** Dynamic shared memory bytes the launch must request; 0 when the
      kernel's shared window fits the 48 KiB static limit. *)
  dynamic_shared_bytes : int;
  (** Total shared memory the kernel's allocations occupy. *)
  shared_bytes : int;
  (** CTA width the kernel fixed, when it fixed one. *)
  threads_per_cta : int option;
}

val compile : Ir.kernel -> compiled

(** [compile_ptx k] is [(compile k).ptx]. *)
val compile_ptx : Ir.kernel -> string
