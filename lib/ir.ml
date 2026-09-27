open Gpu_type
open Mode

type value = {
  id : int; ty : ty; ownership : ownership; locality : locality;
  domain_portability : domain_portability; gpu_boundary : gpu_boundary;
  permission : permission; layout : Layout.t option;
  (* Root buffer identity is provenance for alias analysis. It is not an
     exclusivity claim about this particular derived pointer. *)
  provenance : int option;
}
type instr =
  | Const_i32 of value * int | Const_f32 of value * float
  | Add_i32 of value * value * value | Add_f32 of value * value * value
  | Mul_f32 of value * value * value | Thread_idx_x of value
  | Gep_f32 of value * value * value | Load_f32 of value * value
  | Store_f32 of value * value
  | Shared_alloc of value
  | Load_tensor of value * value
  | Store_tensor of value * value
  | Scale_tensor_f32 of value * value * value
  | Warp_reduce_sum_f32 of value * value
  | Store_f32_lane0 of value * value
  | Barrier of execution_level
  | Return of value option
type arg = { name : string; value : value }
type kernel = { name : string; args : arg list; body : instr list }
type actual = { buffer_id : int; actual_ownership : ownership; actual_permission : permission }

let make_value ?(ownership=Aliased) ?(locality=Mode.Local)
    ?(domain_portability=Domain_portability_unspecified)
    ?(gpu_boundary=Boundary_unspecified) ?(permission=Read_only) ?layout
    ?(provenance=None) id ty =
  { id; ty; ownership; locality; domain_portability; gpu_boundary; permission; layout; provenance }
let results = function
  | Const_i32 (v, _) | Const_f32 (v, _) | Thread_idx_x v | Gep_f32 (v, _, _) -> [v]
  | Add_i32 (v, _, _) | Add_f32 (v, _, _) | Mul_f32 (v, _, _) | Load_f32 (v, _)
  | Shared_alloc v | Load_tensor (v, _) -> [v]
  | Scale_tensor_f32 (v, _, _) -> [v]
  | Warp_reduce_sum_f32 (v, _) -> [v]
  | Store_f32 _ | Store_f32_lane0 _ | Store_tensor _ | Barrier _ | Return _ -> []
let operands = function
  | Const_i32 _ | Const_f32 _ | Thread_idx_x _ -> []
  | Gep_f32 (_, p, ix) -> [p; ix]
  | Add_i32 (_, a, b) | Add_f32 (_, a, b) | Mul_f32 (_, a, b) -> [a; b]
  | Load_f32 (_, p) -> [p] | Store_f32 (p, v) -> [p; v]
  | Shared_alloc _ | Barrier _ -> []
  | Load_tensor (_, src) -> [src]
  | Scale_tensor_f32 (_, tensor, scalar) -> [tensor; scalar]
  | Warp_reduce_sum_f32 (_, tensor) -> [tensor]
  | Store_f32_lane0 (ptr, value) -> [ptr; value]
  | Store_tensor (dst, src) -> [dst; src]
  | Return None -> [] | Return (Some v) -> [v]
