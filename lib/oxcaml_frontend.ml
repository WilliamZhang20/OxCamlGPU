(* Imports the stable metadata emitted from OxCaml compiler Typedtrees. *)
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

(* Import the stable, tab-separated output produced by the version-matched
   Typedtree adapter in tools/export_typedtree_modes.ml. *)
let parse_typedtree_metadata source =
  let parse_kind = function
    | "buffer_f32" -> Ptr (F32, Gpu_type.Global)
    | "scalar_f32" -> F32
    | "scalar_i32" -> I32
    | "unit" -> Unit
    | other -> raise (Parse_error ("unsupported typedtree type " ^ other)) in
  let slot kind mode_fields =
    let ownership = ref Aliased and locality = ref Global
    and domain_portability = ref Domain_nonportable and permission = ref Read_write in
    List.iter (function
      | "unique" -> ownership := Unique
      | "aliased" -> ownership := Aliased
      | "local" -> locality := Local
      | "global" -> locality := Global
      | "portable" -> domain_portability := Domain_portable
      | "nonportable" -> domain_portability := Domain_nonportable
      | "read" -> permission := Read_only
      | "write" -> permission := Write_only
      | "read_write" -> permission := Read_write
      | "immutable" -> permission := Immutable
      | "" -> ()
      | other -> raise (Parse_error ("unexpected field in typedtree metadata: " ^ other))) mode_fields;
    { ty=parse_kind kind; ownership= !ownership; locality= !locality;
      domain_portability= !domain_portability; permission= !permission;
      gpu_boundary=Boundary_unspecified } in
  let lines = String.split_on_char '\n' source |> List.map String.trim |> List.filter ((<>) "") in
  let fields line = String.split_on_char '\t' line in
  let format_seen = ref false in
  let name = ref None and args = ref [] and result = ref None in
  List.iter (fun line -> match fields line with
    | ["format"; "1"] ->
        if !format_seen then raise (Parse_error "duplicate metadata format header");
        format_seen := true
    | ["kernel"; kernel_name] -> name := Some kernel_name
    | ["arg"; index; kind; ownership; locality; portability; permission] ->
        let index = try int_of_string index with Failure _ -> raise (Parse_error "invalid argument index") in
        let mode_fields = [ownership; locality; portability; permission] in
        let value_slot = slot kind mode_fields in
        args := (index, value_slot) :: !args
    | ["result"; kind; ownership; locality; portability; permission] ->
        result := Some (slot kind [ownership; locality; portability; permission])
    | ["body"; _; _; _; _] -> ()
    | _ -> raise (Parse_error ("malformed typedtree metadata line: " ^ line))) lines;
  if not !format_seen then raise (Parse_error "typedtree metadata has no supported format header");
  let name = match !name with Some n -> n | None -> raise (Parse_error "typedtree metadata has no kernel declaration") in
  let args = List.sort (fun (i,_) (j,_) -> compare i j) !args in
  List.iteri (fun expected (actual,_) -> if actual <> expected then raise (Parse_error "typedtree argument indices are not contiguous")) args;
  let args = List.map (fun (id, slot) ->
    { name="arg" ^ string_of_int id; value=make_value ~ownership:slot.ownership
        ~locality:slot.locality ~domain_portability:slot.domain_portability
        ~gpu_boundary:slot.gpu_boundary ~permission:slot.permission id slot.ty }) args in
  let result = match !result with Some r -> r | None -> raise (Parse_error "typedtree metadata has no result") in
  { name; args; result }
