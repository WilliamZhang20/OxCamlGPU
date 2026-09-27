(* Imports the stable metadata emitted from OxCaml compiler Typedtrees. *)
open Gpu_type
open Mode
open Ir

type slot = Gpu_metadata.slot = {
  ty : ty;
  ownership : ownership;
  locality : locality;
  domain_portability : domain_portability;
  gpu_boundary : gpu_boundary;
  permission : permission;
}
type signature = { name : string; args : arg list; result : slot }
exception Parse_error of string

let parse_typedtree_metadata source =
  try
    let metadata = Gpu_metadata.parse source in
    let args = List.map (fun (argument : Gpu_metadata.argument) ->
      let id = argument.index and slot = argument.slot in
      let provenance = match slot.ty with Ptr _ | MemRef _ -> Some id | _ -> None in
      { name="arg" ^ string_of_int id; value=make_value ~ownership:slot.ownership
          ~locality:slot.locality ~domain_portability:slot.domain_portability
          ~gpu_boundary:slot.gpu_boundary
          ~permission:slot.permission ~provenance id slot.ty }) metadata.args in
    { name=metadata.name; args; result=metadata.result }
  with Gpu_metadata.Parse_error message -> raise (Parse_error message)
