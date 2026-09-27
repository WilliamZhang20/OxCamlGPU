type addr_space = Global | Shared | Local

type ty = I32 | F32 | Unit | Ptr of ty * addr_space

type ownership = Unique | Aliased
type locality = Local | Global
(* OxCaml portability is about moving values between domains. It says nothing
   about host/device transferability. *)
type domain_portability = Domain_portable | Domain_nonportable | Domain_portability_unspecified
type gpu_boundary = Boundary_unspecified | Boundary_portable | Boundary_local
type permission = Read_only | Write_only | Read_write | Immutable

let rec string_of_ty = function
  | I32 -> "i32"
  | F32 -> "f32"
  | Unit -> "unit"
  | Ptr (t, space) ->
      let space = match space with Global -> "global" | Shared -> "shared" | Local -> "local" in
      Printf.sprintf "ptr<%s,%s>" (string_of_ty t) space
