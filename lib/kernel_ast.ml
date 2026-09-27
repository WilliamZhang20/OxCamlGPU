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
