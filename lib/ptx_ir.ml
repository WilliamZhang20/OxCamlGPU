(* Legalized target operations. No semantic Ir.instr is carried across this
   boundary: unsupported semantic operations must fail during lowering. *)
type operation =
  | Const_bool of Ir.value * bool
  | Sub_i32 of Ir.value * Ir.value * Ir.value
  | Mul_i32 of Ir.value * Ir.value * Ir.value
  | Compare of Ir.value * Gpu_type.comparison * Ir.value * Ir.value
  | Label of int
  | Jump of int
  | Branch of Ir.value * int * int
  | Move of Ir.value * Ir.value
  | Const_i32 of Ir.value * int
  | Const_f32 of Ir.value * float
  | Add_i32 of Ir.value * Ir.value * Ir.value
  | Add_f32 of Ir.value * Ir.value * Ir.value
  | Mul_f32 of Ir.value * Ir.value * Ir.value
  | Tensor_scale_f32 of Ir.value * Ir.value * Ir.value
  | Tensor_mul_f32 of Ir.value * Ir.value * Ir.value
  | Thread_idx_x of Ir.value
  | Global_idx_x of Ir.value
  | Gep_f32 of Ir.value * Ir.value * Ir.value
  | Load_f32 of Ir.value * Ir.value
  | Load_f32_masked of Ir.value * Ir.value * Ir.value * Ir.value
  | Store_f32 of Ir.value * Ir.value
  | Store_f32_masked of Ir.value * Ir.value * Ir.value * Ir.value
  | Shared_alloc of Ir.value
  | Tile_load_f32 of Ir.value * Ir.value
  | Tile_store_f32 of Ir.value * Ir.value
  | Store_grid_leader_f32 of Ir.value * Ir.value
  | Barrier_cta
  | Tensor_lane_f32 of Ir.value * Ir.value
  | Warp_butterfly_sum_f32 of Ir.value * Ir.value
  | Return

type launch_contract = {
  threads_per_cta : (int * int * int) option;
  static_shared_bytes : int;
}

type kernel = {
  name : string;
  args : Ir.arg list;
  body : operation list;
  launch : launch_contract;
}

let results = function
  | Const_bool(v,_) | Sub_i32(v,_,_) | Mul_i32(v,_,_) | Compare(v,_,_,_) | Move(v,_) | Shared_alloc v -> [v]
  | Label _ | Jump _ | Branch _ -> []
  | Const_i32 (v, _) | Const_f32 (v, _) | Thread_idx_x v | Global_idx_x v
  | Gep_f32 (v, _, _) | Load_f32 (v, _) | Load_f32_masked (v, _, _, _)
  | Tile_load_f32 (v, _) | Tensor_lane_f32 (v, _) | Warp_butterfly_sum_f32 (v, _) -> [v]
  | Add_i32 (v, _, _) | Add_f32 (v, _, _) | Mul_f32 (v, _, _)
  | Tensor_scale_f32 (v, _, _) | Tensor_mul_f32 (v, _, _) -> [v]
  | Store_f32 _ | Store_f32_masked _ | Tile_store_f32 _
  | Store_grid_leader_f32 _ | Barrier_cta | Return -> []

let operands = function
  | Const_bool _ | Label _ | Jump _ -> []
  | Sub_i32(_,a,b) | Mul_i32(_,a,b) | Compare(_,_,a,b) -> [a;b]
  | Branch(v,_,_) | Move(_,v) -> [v]
  | Const_i32 _ | Const_f32 _ | Thread_idx_x _ | Global_idx_x _ | Shared_alloc _
  | Barrier_cta | Return -> []
  | Add_i32 (_, a, b) | Add_f32 (_, a, b) | Mul_f32 (_, a, b)
  | Tensor_scale_f32 (_, a, b) | Tensor_mul_f32 (_, a, b) -> [a;b]
  | Gep_f32 (_, p, i) -> [p;i]
  | Load_f32 (_, p) | Tile_load_f32 (_, p) | Tensor_lane_f32 (_, p)
  | Warp_butterfly_sum_f32 (_, p) -> [p]
  | Load_f32_masked (_, p, i, n) -> [p;i;n]
  | Store_f32 (p, v) | Tile_store_f32 (p, v) | Store_grid_leader_f32 (p, v) -> [p;v]
  | Store_f32_masked (p, v, i, n) -> [p;v;i;n]

let all_values kernel =
  List.concat_map (fun instruction -> operands instruction @ results instruction) kernel.body
  @ List.map (fun arg -> arg.Ir.value) kernel.args
