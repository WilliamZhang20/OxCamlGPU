(* Kernel_ast is the frontend-facing name for the versioned interchange AST.
   The rows are parsed once by Gpu_metadata and remain typed from here on. *)
type scalar = Gpu_metadata.scalar = I32 | F32
type arg_kind = Gpu_metadata.arg_kind = Buffer of scalar | Scalar of scalar
type atom = Gpu_metadata.atom = Arg of int | Value of int
type memory = Gpu_metadata.memory = Buffer_arg of int | Pointer of atom
type operation = Gpu_metadata.operation =
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

type t = { name : string; args : arg_kind list; body : operation list }
exception Parse_error of string

let of_compiler_metadata source =
  let metadata = try Gpu_metadata.parse source with
    | Gpu_metadata.Parse_error message -> raise (Parse_error message) in
  let args = List.map (fun (arg : Gpu_metadata.argument) ->
    match arg.slot.ty with
    | Gpu_type.MemRef (_, Gpu_type.Float32, _) -> Buffer F32
    | Gpu_type.F32 -> Scalar F32
    | Gpu_type.I32 -> Scalar I32
    | _ -> raise (Parse_error "unsupported typedtree argument type")) metadata.args in
  { name=metadata.name; args; body=metadata.body }
