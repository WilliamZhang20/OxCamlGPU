type ownership = Unique | Aliased
type locality = Local | Global

(* OxCaml portability is about moving values between domains. It says nothing
   about host/device transferability. *)
type domain_portability = Domain_portable | Domain_nonportable | Domain_portability_unspecified
type gpu_boundary = Boundary_unspecified | Boundary_portable | Boundary_local
type permission = Read_only | Write_only | Read_write | Immutable
