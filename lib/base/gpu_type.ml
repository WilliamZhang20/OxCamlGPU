(* GPU IR types: scalars, pointers, tensors, and addressable MemRefs. *)
type addr_space = Global | Shared | Local

(* Element types of a buffer or tile. Width is a property of the type, so
   address arithmetic asks [dtype_storage_bytes] rather than writing 4. *)
type dtype = Int32 | Float32 | Bfloat16 | Float16
type dim = Static of int | Symbol of string | Dynamic
type shape = dim list

type ty =
  | Bool
  | I32
  | U64
  | F32
  | Unit
  | Ptr of ty * addr_space
  | Tensor of shape * dtype
  | MemRef of shape * dtype * addr_space
  (* Opaque Hopper CUtensorMap (128B); passed as a device pointer to the map. *)
  | Tensor_map

let dtype_storage_bytes = function
  | Int32 | Float32 -> 4
  | Bfloat16 | Float16 -> 2

(* The f32 register type every narrow element converts through. There is no
   bf16 or f16 arithmetic: those are storage formats that TMA moves and WGMMA
   consumes from shared memory, which is also how a CUTLASS bf16 GEMM works. *)
let is_narrow_float = function Bfloat16 | Float16 -> true | Int32 | Float32 -> false
let is_float_dtype = function Float32 | Bfloat16 | Float16 -> true | Int32 -> false

let shape_elements shape =
  List.fold_left (fun n -> function
    | Static extent when extent > 0 && n <= max_int / extent -> n * extent
    | Static _ -> invalid_arg "shape storage size overflows the host integer range"
    | Symbol _ | Dynamic -> invalid_arg "storage size requires a fully static shape") 1 shape

let storage_bytes shape dtype =
  let elements = shape_elements shape and element_bytes = dtype_storage_bytes dtype in
  if elements > max_int / element_bytes then
    invalid_arg "storage byte size overflows the host integer range";
  elements * element_bytes

let storage_bytes_of_ty = function
  | Tensor (shape, dtype) | MemRef (shape, dtype, _) -> storage_bytes shape dtype
  | _ -> invalid_arg "storage size is undefined for this type"

let string_of_dim = function
  | Static n -> string_of_int n
  | Symbol name -> name
  | Dynamic -> "?"

let string_of_shape shape =
  "[" ^ String.concat "," (List.map string_of_dim shape) ^ "]"

let string_of_dtype = function
  | Int32 -> "i32" | Float32 -> "f32" | Bfloat16 -> "bf16" | Float16 -> "f16"

let rec string_of_ty = function
  | Bool -> "bool"
  | I32 -> "i32"
  | U64 -> "u64"
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
  | Tensor_map -> "tensor_map"

let is_global_f32_memref = function
  | MemRef ([_], Float32, Global) -> true
  | _ -> false

let is_static_global_f32_tile = function
  | MemRef (shape, Float32, Global) ->
      List.length shape <= 2 &&
      List.for_all (function Static n -> n > 0 | _ -> false) shape
  | _ -> false

let is_global_f32_buffer = function
  | Ptr (F32, Global) | MemRef (_, Float32, Global) -> true
  | _ -> false

(* Float intrinsics: one PTX instruction each, pure, and opaque to algebraic
   reasoning — passes may eliminate a redundant one but never rewrite it.
   [Add_f32], [Mul_f32] and [Mad_f32] stay their own instructions because
   passes do reason about those (commutative CSE, the tensor path); this is
   the line LLVM draws between [fadd] and an [llvm.*] intrinsic.

   [_approx] names the hardware's fast approximate unit. Choosing it is a
   numerical decision, so it is in the name rather than inferred: a kernel
   that wants IEEE rounding writes [sqrt] or [/.] and gets [sqrt.rn]/[div.rn]. *)
type unary_f32 =
  | Negate
  | Absolute
  | Sqrt
  | Rsqrt_approx
  | Exp2_approx
  | Log2_approx
  | Tanh_approx
  | Recip_approx

type binary_f32 = Subtract | Divide | Maximum | Minimum

let string_of_unary_f32 = function
  | Negate -> "neg" | Absolute -> "abs" | Sqrt -> "sqrt"
  | Rsqrt_approx -> "rsqrt_approx" | Exp2_approx -> "exp2_approx"
  | Log2_approx -> "log2_approx" | Tanh_approx -> "tanh_approx"
  | Recip_approx -> "recip_approx"

let unary_f32_of_string = function
  | "neg" -> Some Negate | "abs" -> Some Absolute | "sqrt" -> Some Sqrt
  | "rsqrt_approx" -> Some Rsqrt_approx | "exp2_approx" -> Some Exp2_approx
  | "log2_approx" -> Some Log2_approx | "tanh_approx" -> Some Tanh_approx
  | "recip_approx" -> Some Recip_approx
  | _ -> None

let string_of_binary_f32 = function
  | Subtract -> "sub" | Divide -> "div" | Maximum -> "max" | Minimum -> "min"

let binary_f32_of_string = function
  | "sub" -> Some Subtract | "div" -> Some Divide
  | "max" -> Some Maximum | "min" -> Some Minimum
  | _ -> None

type comparison = Eq | Ne | Lt | Le | Gt | Ge

let string_of_comparison = function Eq->"eq" | Ne->"ne" | Lt->"lt" | Le->"le" | Gt->"gt" | Ge->"ge"
