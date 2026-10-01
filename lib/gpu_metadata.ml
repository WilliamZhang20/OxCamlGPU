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
type scalar = I32 | F32
type arg_kind = Buffer of scalar | Scalar of scalar
type atom = Arg of int | Value of int
type memory = Buffer_arg of int | Pointer of atom
type operation =
  | Const_i32 of int * int
  | Const_f32 of int * float
  | Thread_idx_x of int
  | Global_idx_x of int
  | Gep_f32 of int * memory * atom
  | Load_f32 of int * atom
  | Load_f32_masked of int * atom * atom * atom
  | Add_f32 of int * atom * atom
  | Mul_f32 of int * atom * atom
  | Warp_reduce_sum_f32 of int * atom
  | Store_f32 of atom * atom
  | Store_f32_masked of atom * atom * atom * atom
  | Store_grid_leader_f32 of memory * atom
type t = { name : string; args : argument list; result : slot; body : operation list }
exception Parse_error of string

let parse_int text = try int_of_string text with Failure _ -> raise (Parse_error ("invalid integer " ^ text))
let parse_atom text = match String.split_on_char ':' text with
  | ["arg"; index] -> Arg (parse_int index)
  | ["val"; index] -> Value (parse_int index)
  | _ -> raise (Parse_error ("invalid SSA operand " ^ text))
let parse_memory text = match String.split_on_char ':' text with
  | ["buf"; index] -> Buffer_arg (parse_int index)
  | ["val"; _] -> Pointer (parse_atom text)
  | _ -> raise (Parse_error ("invalid memory operand " ^ text))
let parse_operation line = match String.split_on_char '\t' line with
  | ["body"; "const_i32"; dst; n] -> Const_i32 (parse_int dst, parse_int n)
  | ["body"; "const_f32"; dst; n] ->
      (try Const_f32 (parse_int dst, float_of_string n)
       with Failure _ -> raise (Parse_error ("invalid f32 literal " ^ n)))
  | ["body"; "thread_idx_x"; dst] -> Thread_idx_x (parse_int dst)
  | ["body"; "global_idx_x"; dst] -> Global_idx_x (parse_int dst)
  | ["body"; "gep_f32"; dst; base; index] -> Gep_f32 (parse_int dst, parse_memory base, parse_atom index)
  | ["body"; "load_f32"; dst; ptr] -> Load_f32 (parse_int dst, parse_atom ptr)
  | ["body"; "load_f32_masked"; dst; ptr; index; bound] ->
      Load_f32_masked (parse_int dst, parse_atom ptr, parse_atom index, parse_atom bound)
  | ["body"; "add_f32"; dst; a; b] -> Add_f32 (parse_int dst, parse_atom a, parse_atom b)
  | ["body"; "mul_f32"; dst; a; b] -> Mul_f32 (parse_int dst, parse_atom a, parse_atom b)
  | ["body"; "warp_reduce_sum_f32"; dst; src] -> Warp_reduce_sum_f32 (parse_int dst, parse_atom src)
  | ["body"; "store_f32"; ptr; value] -> Store_f32 (parse_atom ptr, parse_atom value)
  | ["body"; "store_f32_masked"; ptr; value; index; bound] ->
      Store_f32_masked (parse_atom ptr, parse_atom value, parse_atom index, parse_atom bound)
  | ["body"; "store_grid_leader_f32"; ptr; value] -> Store_grid_leader_f32 (parse_memory ptr, parse_atom value)
  | _ -> raise (Parse_error ("malformed typed operation row: " ^ line))

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
    | ["format"; "2"] -> if !format_seen then raise (Parse_error "duplicate metadata format header") else format_seen := true
    | ["format"; version] -> raise (Parse_error ("unsupported metadata version " ^ version))
    | ["kernel"; n] -> if !name <> None then raise (Parse_error "duplicate kernel declaration") else name := Some n
    | ["arg"; index; kind; own; loc; port; perm] ->
        let index = try int_of_string index with Failure _ -> raise (Parse_error "invalid argument index") in
        args := { index; slot=slot kind [own;loc;port;perm] } :: !args
    | ["result"; kind; own; loc; port; perm] ->
        if !result <> None then raise (Parse_error "duplicate result declaration");
        result := Some (slot kind [own;loc;port;perm])
    | "body" :: _ -> body := parse_operation line :: !body
    | _ -> raise (Parse_error ("malformed typedtree metadata line: " ^ line))) lines;
  if not !format_seen then raise (Parse_error "typedtree metadata has no supported format header");
  let name = match !name with Some n -> n | None -> raise (Parse_error "typedtree metadata has no kernel declaration") in
  let args = List.sort (fun a b -> compare a.index b.index) !args in
  List.iteri (fun i arg -> if arg.index <> i then raise (Parse_error "typedtree argument indices are not contiguous")) args;
  let result = match !result with Some r -> r | None -> raise (Parse_error "typedtree metadata has no result") in
  { name; args; result; body=List.rev !body }
