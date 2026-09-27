type scalar = I32 | F32
type arg_kind = Buffer of scalar | Scalar of scalar

type expr =
  | Arg of int
  | Thread_idx_x
  | Load_f32 of int * expr
  | Add_f32 of expr * expr
  | Mul_f32 of expr * expr
  | I32_const of int
  | F32_const of float

type stmt = Store_f32 of int * expr * expr
type t = { name : string; args : arg_kind list; body : stmt list }

exception Parse_error of string

let of_compiler_metadata source =
  let fields = String.split_on_char '\n' source |> List.filter ((<>) "") |> List.map (String.split_on_char '\t') in
  let name = ref None and args = ref [] and body = ref [] in
  let kind = function
    | "buffer_f32" -> Buffer F32
    | "scalar_f32" -> Scalar F32
    | "scalar_i32" -> Scalar I32
    | other -> raise (Parse_error ("unsupported typedtree argument " ^ other)) in
  let rec expr tokens = match tokens with
    | "thread_idx_x" :: rest -> Thread_idx_x, rest
    | "arg" :: index :: rest -> Arg (int_of_string index), rest
    | "load" :: index :: rest -> let index_expr, rest = expr rest in Load_f32 (int_of_string index,index_expr),rest
    | "add" :: rest -> let a, rest=expr rest in let b, rest=expr rest in Add_f32(a,b),rest
    | "mul" :: rest -> let a, rest=expr rest in let b, rest=expr rest in Mul_f32(a,b),rest
    | _ -> raise (Parse_error "malformed typedtree expression") in
  List.iter (function
    | ["kernel"; kernel_name] -> name := Some kernel_name
    | ["arg"; index; argument_kind; _ownership; _locality; _portability; _permission] ->
        args := (int_of_string index, kind argument_kind) :: !args
    | ["body"; "store"; buffer; index; value] ->
        let parse_expr source =
          let tokens=String.split_on_char ' ' source |> List.filter ((<>) "") in
          let result, rest=expr tokens in
          if rest <> [] then raise (Parse_error "trailing tokens in typedtree expression"); result in
        body := Store_f32(int_of_string buffer,parse_expr index,parse_expr value) :: !body
    | ["result"; _; _; _; _; _] -> ()
    | _ -> raise (Parse_error "malformed Typedtree metadata")) fields;
  let name=match !name with Some name->name | None->raise(Parse_error "missing kernel name") in
  let args=List.sort (fun (i,_) (j,_) -> compare i j) !args in
  List.iteri (fun expected (actual,_) -> if expected<>actual then raise(Parse_error "noncontiguous argument indices")) args;
  { name; args=List.map snd args; body=List.rev !body }
