open Gpu_type
open Mode

type slot = {
  ty : ty;
  ownership : ownership;
  locality : locality;
  domain_portability : domain_portability;
  gpu_boundary : gpu_boundary;
  permission : permission;
}
type argument = { index : int; slot : slot }
type t = { name : string; args : argument list; result : slot; body : string list }
exception Parse_error of string

let parse source =
  let parse_ty = function
    | "buffer_f32" -> MemRef ([Dynamic], Float32, Global)
    | "scalar_f32" -> F32 | "scalar_i32" -> I32 | "unit" -> Unit
    | other -> raise (Parse_error ("unsupported typedtree type " ^ other)) in
  let slot kind modes =
    let ownership = ref None and locality = ref None
    and portability = ref None and permission = ref None in
    let set field value = match !field with
      | None -> field := Some value
      | Some old when old = value -> raise (Parse_error "duplicate mode in typedtree metadata")
      | Some _ -> raise (Parse_error "conflicting modes in typedtree metadata") in
    List.iter (function
      | "unique" -> set ownership Unique | "aliased" -> set ownership Aliased
      | "local" -> set locality Local | "global" -> set locality Global
      | "portable" -> set portability Domain_portable | "nonportable" -> set portability Domain_nonportable
      | "read" -> set permission Read_only | "write" -> set permission Write_only
      | "read_write" -> set permission Read_write | "immutable" -> set permission Immutable
      | "" -> () | other -> raise (Parse_error ("unexpected mode in typedtree metadata: " ^ other))) modes;
    { ty=parse_ty kind; ownership=Option.value !ownership ~default:Aliased;
      locality=Option.value !locality ~default:Global;
      domain_portability=Option.value !portability ~default:Domain_nonportable;
      gpu_boundary=Boundary_unspecified;
      permission=Option.value !permission ~default:Read_write } in
  let format_seen = ref false and name = ref None and args = ref []
  and result = ref None and body = ref [] in
  let lines = String.split_on_char '\n' source |> List.map String.trim |> List.filter ((<>) "") in
  List.iter (fun line -> match String.split_on_char '\t' line with
    | ["format"; "1"] -> if !format_seen then raise (Parse_error "duplicate metadata format header") else format_seen := true
    | ["kernel"; n] -> if !name <> None then raise (Parse_error "duplicate kernel declaration") else name := Some n
    | ["arg"; index; kind; own; loc; port; perm] ->
        let index = try int_of_string index with Failure _ -> raise (Parse_error "invalid argument index") in
        args := { index; slot=slot kind [own;loc;port;perm] } :: !args
    | ["result"; kind; own; loc; port; perm] ->
        if !result <> None then raise (Parse_error "duplicate result declaration");
        result := Some (slot kind [own;loc;port;perm])
    | "body" :: _ -> body := line :: !body
    | _ -> raise (Parse_error ("malformed typedtree metadata line: " ^ line))) lines;
  if not !format_seen then raise (Parse_error "typedtree metadata has no supported format header");
  let name = match !name with Some n -> n | None -> raise (Parse_error "typedtree metadata has no kernel declaration") in
  let args = List.sort (fun a b -> compare a.index b.index) !args in
  List.iteri (fun i arg -> if arg.index <> i then raise (Parse_error "typedtree argument indices are not contiguous")) args;
  let result = match !result with Some r -> r | None -> raise (Parse_error "typedtree metadata has no result") in
  { name; args; result; body=List.rev !body }
