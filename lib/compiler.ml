(* One pass structure, two results: the PTX text and the launch facts a
   caller cannot recover from the text alone. *)
type compiled = {
  ptx : string;
  (* Dynamic shared memory the launch must request; 0 when static. *)
  dynamic_shared_bytes : int;
  (* Total shared memory the kernel's allocations occupy. *)
  shared_bytes : int;
  (* CTA width the kernel was compiled for, when it fixed one. *)
  threads_per_cta : int option;
}

let compile kernel =
  let kernel = Uniformity.promote kernel in
  let kernel = Optimizer.optimize kernel in
  (* Coalesce adjacent scalar stores into vector stores where provably legal.
     Verify afterwards: the pass rewrites memory operations. *)
  let kernel = Store_vector.fuse kernel in
  Verifier.verify_exn kernel;
  let target = Ptx_lowering.lower kernel in
  let physical = Physical_ir.physicalize target in
  let shared = Ptx.shared_requirement physical in
  { ptx = Ptx.emit_physical physical;
    dynamic_shared_bytes = shared.Ptx.dynamic_bytes;
    shared_bytes = shared.Ptx.total_bytes;
    threads_per_cta = kernel.Ir.threads_per_cta }

let compile_ptx kernel = (compile kernel).ptx
