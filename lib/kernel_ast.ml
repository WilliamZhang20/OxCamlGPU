open Gpu_type
open Gpu_mode

type slot = {
  loc : Source_span.t;
  ty : ty;
  ownership : ownership;
  locality : locality;
  domain_portability : domain_portability;
  gpu_boundary : gpu_boundary;
  permission : permission;
}
type argument = { index : int; slot : slot }
type atom = Arg of int | Value of int
type memory = Buffer_arg of int | Pointer of atom
type operation =
  | Const_bool of int * bool
  | Add_i32 of int * atom * atom
  | Sub_i32 of int * atom * atom
  | Mul_i32 of int * atom * atom
  | Compare of int * comparison * atom * atom
  | If of (int * ty) option * atom * region * region
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
and instruction = { op : operation; loc : Source_span.t }
and region = { body : instruction list; yield : atom option }
type t = { name : string; args : argument list; result : slot; body : instruction list }

let instruction ?(loc=Source_span.synthetic) op = {op;loc}
