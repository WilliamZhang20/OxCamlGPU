type addr_space = Global | Shared | Local

type dtype = Int32 | Float32
type dim = Static of int | Symbol of string | Dynamic
type shape = dim list

type ty =
  | Bool
  | I32
  | F32
  | Unit
  | Ptr of ty * addr_space
  | Tensor of shape * dtype
  | MemRef of shape * dtype * addr_space

let string_of_dim = function
  | Static n -> string_of_int n
  | Symbol name -> name
  | Dynamic -> "?"

let string_of_shape shape =
  "[" ^ String.concat "," (List.map string_of_dim shape) ^ "]"

let string_of_dtype = function Int32 -> "i32" | Float32 -> "f32"

let rec string_of_ty = function
  | Bool -> "bool"
  | I32 -> "i32"
  | F32 -> "f32"
  | Unit -> "unit"
  | Ptr (t, space) ->
      let space = match space with Global -> "global" | Shared -> "shared" | Local -> "local" in
      Printf.sprintf "ptr<%s,%s>" (string_of_ty t) space
  | Tensor (shape, dtype) ->
      Printf.sprintf "tensor<%s,%s>" (string_of_shape shape) (string_of_dtype dtype)
  | MemRef (shape, dtype, space) ->
      let space = match space with Global -> "global" | Shared -> "shared" | Local -> "local" in
      Printf.sprintf "memref<%s,%s,%s>" (string_of_shape shape) (string_of_dtype dtype) space

let is_global_f32_memref = function
  | MemRef ([_], Float32, Global) -> true
  | _ -> false

let is_global_f32_buffer = function
  | Ptr (F32, Global) | MemRef (_, Float32, Global) -> true
  | _ -> false

type comparison = Eq | Ne | Lt | Le | Gt | Ge

let string_of_comparison = function Eq->"eq" | Ne->"ne" | Lt->"lt" | Le->"le" | Gt->"gt" | Ge->"ge"
