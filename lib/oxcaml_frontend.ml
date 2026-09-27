(* A deliberately small importer for OxCaml val declarations. It does not
   parse OCaml generally or inspect compiler artifacts; see README. *)
open Gpu_type
open Mode
open Ir

type slot = {
  ty : ty;
  ownership : ownership;
  locality : locality;
  domain_portability : domain_portability;
  permission : permission;
  gpu_boundary : gpu_boundary;
}
type signature = { name : string; args : arg list; result : slot }
exception Parse_error of string

let trim = String.trim
let split_once s delimiter =
  match String.index_opt s delimiter with
  | None -> None
  | Some i -> Some (trim (String.sub s 0 i), trim (String.sub s (i + 1) (String.length s - i - 1)))

let parse_type annotation =
  let base, modes = match split_once annotation '@' with
    | None -> trim annotation, []
    | Some (base, modes) -> base, String.split_on_char ' ' modes |> List.map trim |> List.filter ((<>) "") in
  let ownership = ref Aliased and locality = ref Global and domain_portability = ref Domain_nonportable
  and permission = ref Read_write in
  let unique_seen = ref false and locality_seen = ref false and portability_seen = ref false
  and permission_seen = ref false in
  List.iter (function
    | "unique" -> if !unique_seen then raise (Parse_error "repeated uniqueness mode"); unique_seen := true; ownership := Unique
    | "aliased" -> if !unique_seen then raise (Parse_error "conflicting uniqueness modes"); unique_seen := true; ownership := Aliased
    | "local" -> if !locality_seen then raise (Parse_error "conflicting locality modes"); locality_seen := true; locality := Local
    | "global" -> if !locality_seen then raise (Parse_error "conflicting locality modes"); locality_seen := true; locality := Global
    | "portable" -> if !portability_seen then raise (Parse_error "conflicting portability modes"); portability_seen := true; domain_portability := Domain_portable
    | "nonportable" -> if !portability_seen then raise (Parse_error "conflicting portability modes"); portability_seen := true; domain_portability := Domain_nonportable
    | "read" -> if !permission_seen then raise (Parse_error "conflicting visibility modes"); permission_seen := true; permission := Read_only
    | "write" -> if !permission_seen then raise (Parse_error "conflicting visibility modes"); permission_seen := true; permission := Write_only
    | "read_write" -> if !permission_seen then raise (Parse_error "conflicting visibility modes"); permission_seen := true; permission := Read_write
    | "immutable" -> if !permission_seen then raise (Parse_error "conflicting visibility modes"); permission_seen := true; permission := Immutable
    | mode -> raise (Parse_error ("unsupported mode @ " ^ mode))) modes;
  let ty = match base with
    | "f32 gpu_array" | "float gpu_array" -> Ptr (F32, Gpu_type.Global)
    | "f32" | "float" -> F32
    | "i32" | "int" -> I32
    | "unit" -> Unit
    | other -> raise (Parse_error ("unsupported type " ^ other)) in
  { ty; ownership= !ownership; locality= !locality; domain_portability= !domain_portability;
    permission= !permission; gpu_boundary=Boundary_unspecified }

let parse_val_line line =
  let line = trim line in
  if not (String.starts_with ~prefix:"val " line) then None
  else match split_once (String.sub line 4 (String.length line - 4)) ':' with
    | None -> raise (Parse_error ("expected `val name : type`: " ^ line))
    | Some (name, typ) ->
        let rec split_arrows remaining acc =
          match String.index_opt remaining '-' with
          | Some i when i + 1 < String.length remaining && remaining.[i+1] = '>' ->
              let lhs = trim (String.sub remaining 0 i) in
              let rhs = trim (String.sub remaining (i+2) (String.length remaining-i-2)) in
              split_arrows rhs (lhs :: acc)
          | _ -> List.rev (trim remaining :: acc) in
        let parts = split_arrows typ [] in
        if name = "" || List.length parts < 2 then raise (Parse_error ("expected an arrow type in: " ^ line));
        let args, result = List.rev (List.tl (List.rev parts)), List.hd (List.rev parts) in
        let parsed_args = List.mapi (fun i s ->
          let label, type_s = match split_once s ':' with Some (label, t) -> label, t | None -> "arg" ^ string_of_int i, s in
          let slot = parse_type type_s in
          { name=label; value=make_value ~ownership:slot.ownership ~locality:slot.locality
              ~domain_portability:slot.domain_portability ~gpu_boundary:slot.gpu_boundary
              ~permission:slot.permission i slot.ty }) args in
        let result = parse_type result in
        Some { name; args=parsed_args; result }

let parse_string source =
  let lines = String.split_on_char '\n' source in
  let rec gather lines current declarations = match lines with
    | [] ->
        let declarations = match current with
          | None -> declarations
          | Some line -> (match parse_val_line line with Some s -> s :: declarations | None -> declarations) in
        List.rev declarations
    | raw :: rest ->
        let line = trim raw in
        if line = "" || String.starts_with ~prefix:"(*" line || String.starts_with ~prefix:"type " line then
          gather rest current declarations
        else if String.starts_with ~prefix:"val " line then begin
          let declarations = match current with
            | None -> declarations
            | Some old -> (match parse_val_line old with Some s -> s :: declarations | None -> declarations) in
          if String.ends_with ~suffix:"->" line then gather rest (Some line) declarations
          else gather rest (Some line) declarations
        end else match current with
          | None -> gather rest None declarations
          | Some old ->
              let joined = old ^ " " ^ line in
              if String.ends_with ~suffix:"->" line then gather rest (Some joined) declarations
              else gather rest (Some joined) declarations in
  let declarations = gather lines None [] in
  match declarations with
  | [signature] -> signature
  | [] -> raise (Parse_error "no `val` declaration found")
  | _ -> raise (Parse_error "expected exactly one `val` declaration")

let parse_file path =
  let channel = open_in path in
  Fun.protect ~finally:(fun () -> close_in channel) (fun () ->
    let source = really_input_string channel (in_channel_length channel) in
    parse_string source)

let as_kernel ~body signature = { name=signature.name; args=signature.args; body }
