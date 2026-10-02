let string_of_ownership = function Gpu_mode.Unique -> "unique" | Aliased -> "aliased"
let string_of_locality = function Gpu_mode.Local -> "local" | Global -> "global"
let string_of_domain_portability = function
  | Gpu_mode.Domain_portable -> "portable"
  | Domain_nonportable -> "nonportable"
  | Domain_portability_unspecified -> "unspecified"
let string_of_boundary = function
  | Gpu_mode.Boundary_unspecified -> "unspecified"
  | Boundary_portable -> "portable"
  | Boundary_local -> "local"
let string_of_permission = function
  | Gpu_mode.Read_only -> "read"
  | Write_only -> "write"
  | Read_write -> "read_write"
  | Immutable -> "immutable"

let () =
  if Array.length Sys.argv <> 2 then begin
    prerr_endline "usage: import_signature.exe FILE.gpu";
    exit 2
  end;
  let channel = open_in Sys.argv.(1) in
  let metadata = Fun.protect ~finally:(fun () -> close_in channel) (fun () ->
    really_input_string channel (in_channel_length channel)) in
  let signature = Oxcaml_frontend.import metadata in
  Printf.printf "kernel %s\n" signature.name;
  List.iteri (fun i arg ->
    let value = arg.Kernel_ast.slot in
    Printf.printf "  arg%d (%s): type=%s ownership=%s locality=%s domain_portability=%s gpu_boundary=%s permission=%s\n"
      i ("arg" ^ string_of_int i) (Gpu_type.string_of_ty value.ty)
      (string_of_ownership value.ownership) (string_of_locality value.locality)
      (string_of_domain_portability value.domain_portability)
      (string_of_boundary value.gpu_boundary) (string_of_permission value.permission))
    signature.args;
  Printf.printf "  result: type=%s ownership=%s locality=%s domain_portability=%s gpu_boundary=%s permission=%s\n"
    (Gpu_type.string_of_ty signature.result.ty)
    (string_of_ownership signature.result.ownership)
    (string_of_locality signature.result.locality)
    (string_of_domain_portability signature.result.domain_portability)
    (string_of_boundary signature.result.gpu_boundary)
    (string_of_permission signature.result.permission)
