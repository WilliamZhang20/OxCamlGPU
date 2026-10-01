type ownership = Unique | Aliased
type locality = Local | Global

(* OxCaml portability is about moving values between domains. It says nothing
   about host/device transferability. *)
type domain_portability = Domain_portable | Domain_nonportable | Domain_portability_unspecified
type gpu_boundary = Boundary_unspecified | Boundary_portable | Boundary_local
type permission = Read_only | Write_only | Read_write | Immutable

let can_read = function Read_only | Read_write | Immutable -> true | Write_only -> false
let can_write = function Write_only | Read_write -> true | Read_only | Immutable -> false

(* An actual may satisfy a formal when it grants every operation the formal
   needs. Immutable additionally promises that no mutable reference exists,
   so it is accepted by read-only formals but required by immutable formals. *)
let satisfies ~required actual =
  (not (can_read required) || can_read actual) &&
  (not (can_write required) || can_write actual) &&
  (required <> Immutable || actual = Immutable)

(* A value that can outlive the region is usable where only a local lifetime
   is required; a region-local value cannot satisfy a global-lifetime formal. *)
let locality_satisfies ~required actual =
  match required, actual with
  | Local, _ | Global, Global -> true
  | Global, Local -> false

let domain_portability_satisfies ~required actual =
  match required, actual with
  | Domain_portable, Domain_portable -> true
  | Domain_portable, (Domain_nonportable | Domain_portability_unspecified) -> false
  | (Domain_nonportable | Domain_portability_unspecified), _ -> true

let gpu_boundary_satisfies ~required actual =
  match required, actual with
  | Boundary_portable, Boundary_portable -> true
  | Boundary_portable, (Boundary_local | Boundary_unspecified) -> false
  | (Boundary_local | Boundary_unspecified), _ -> true
