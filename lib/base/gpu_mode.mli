(** Modal facts carried on IR values.

    OxCaml domain portability is distinct from GPU host/device boundary
    portability; see [domain_portability] vs [gpu_boundary]. *)

type ownership = Unique | Aliased
type locality = Local | Global

type domain_portability =
  | Domain_portable
  | Domain_nonportable
  | Domain_portability_unspecified

type gpu_boundary =
  | Boundary_unspecified
  | Boundary_portable
  | Boundary_local

type permission = Read_only | Write_only | Read_write | Immutable

val can_read : permission -> bool
val can_write : permission -> bool
val satisfies : required:permission -> permission -> bool
val locality_satisfies : required:locality -> locality -> bool
val domain_portability_satisfies :
  required:domain_portability -> domain_portability -> bool
val gpu_boundary_satisfies : required:gpu_boundary -> gpu_boundary -> bool

val meet_locality : locality -> locality -> locality
val meet_domain_portability :
  domain_portability -> domain_portability -> domain_portability
val meet_gpu_boundary : gpu_boundary -> gpu_boundary -> gpu_boundary
val meet_permission : permission -> permission -> permission
