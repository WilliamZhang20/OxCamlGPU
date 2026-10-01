type scalar = I32 | F32
type arg_kind = Buffer of scalar | Scalar of scalar

type expr =
  | Arg of int
  | Local of int
  | Thread_idx_x
  | Global_idx_x
  | Load_f32 of int * expr
  | Load_f32_masked of int * expr * expr
  | Add_f32 of expr * expr
  | Mul_f32 of expr * expr
  | Warp_sum_f32 of expr
  | I32_const of int
  | F32_const of float

type stmt =
  | Let of int * expr
  | Store_f32 of int * expr * expr
  | Store_f32_masked of int * expr * expr * expr
  | Store_grid_leader_f32 of int * expr
type t = { name : string; args : arg_kind list; body : stmt list }

exception Parse_error of string

let of_compiler_metadata source =
  let rec expr tokens = match tokens with
    | "thread_idx_x" :: rest -> Thread_idx_x, rest
    | "global_idx_x" :: rest -> Global_idx_x, rest
    | "arg" :: index :: rest -> Arg (int_of_string index), rest
    | "local" :: index :: rest -> Local (int_of_string index), rest
    | "i32" :: value :: rest -> I32_const (int_of_string value), rest
    | "f32" :: value :: rest ->
        (try F32_const (float_of_string value), rest
         with Failure _ -> raise (Parse_error ("invalid f32 literal " ^ value)))
    | "load" :: index :: rest -> let index_expr, rest = expr rest in Load_f32 (int_of_string index,index_expr),rest
    | "load_masked" :: index :: rest -> let ix, rest=expr rest in let n, rest=expr rest in Load_f32_masked(int_of_string index,ix,n),rest
    | "add" :: rest -> let a, rest=expr rest in let b, rest=expr rest in Add_f32(a,b),rest
    | "mul" :: rest -> let a, rest=expr rest in let b, rest=expr rest in Mul_f32(a,b),rest
    | "warp_sum" :: rest -> let a, rest=expr rest in Warp_sum_f32 a,rest
    | _ -> raise (Parse_error "malformed typedtree expression") in
  try
  let metadata = Gpu_metadata.parse source in
  let args = List.map (fun (arg : Gpu_metadata.argument) ->
    let arg_kind = match arg.slot.ty with
      | Gpu_type.MemRef (_, Gpu_type.Float32, _) -> Buffer F32
      | Gpu_type.F32 -> Scalar F32 | Gpu_type.I32 -> Scalar I32
      | _ -> raise (Parse_error "unsupported typedtree argument type") in
    arg.index, arg_kind) metadata.args in
  let parse_expr source =
    let tokens=String.split_on_char ' ' source |> List.filter ((<>) "") in
    let result, rest=expr tokens in
    if rest <> [] then raise (Parse_error "trailing tokens in typedtree expression");
    result in
  let body = ref [] in
  List.iter (fun line -> match String.split_on_char '\t' line with
    | ["body"; "let"; local; value] ->
        body := Let(int_of_string local, parse_expr value) :: !body
    | ["body"; "store"; buffer; index; value] ->
        body := Store_f32(int_of_string buffer,parse_expr index,parse_expr value) :: !body
    | ["body"; "store_grid_leader"; buffer; value] ->
        body := Store_grid_leader_f32(int_of_string buffer,parse_expr value) :: !body
    | ["body"; "store_masked"; buffer; index; bound; value] ->
        body := Store_f32_masked(int_of_string buffer,parse_expr index,parse_expr bound,parse_expr value) :: !body
    | _ -> raise (Parse_error "malformed typedtree body")) metadata.body;
  { name=metadata.name; args=List.map (fun arg -> snd arg) args; body=List.rev !body }
  with Gpu_metadata.Parse_error message -> raise (Parse_error message)
