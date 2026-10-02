open Gpu_type
open Gpu_mode

type value = {
  id : int; ty : ty; loc : Source_span.t; ownership : ownership; locality : locality;
  domain_portability : domain_portability; gpu_boundary : gpu_boundary;
  permission : permission; layout : Layout.t option;
  (* Root buffer identity is provenance for alias analysis. It is not an
     exclusivity claim about this particular derived pointer. *)
  provenance : int option;
}

type instr =
  | Const_bool of value * bool
  | Sub_i32 of value * value * value
  | Mul_i32 of value * value * value
  | Compare of value * comparison * value * value
  | If of value option * value * region * region
  | Const_i32 of value * int | Const_f32 of value * float
  | Add_i32 of value * value * value | Add_f32 of value * value * value
  | Mul_f32 of value * value * value | Thread_idx_x of value | Global_idx_x of value
  | Gep_f32 of value * value * value | Load_f32 of value * value
  | Load_f32_masked of value * value * value * value
  | Store_f32 of value * value | Store_f32_masked of value * value * value * value
  | Shared_alloc of value
  | Load_tensor of value * value
  | Store_tensor of value * value
  | Scale_tensor_f32 of value * value * value
  (* Semantic operations retained until target strategy selection. *)
  | Mul_tensor_f32 of value * value * value
  | Reduce_sum_f32 of value * value
  | Warp_sum_f32 of value * value
  | Store_f32_grid_leader of value * value
  | Barrier of Execution.level
  | Return of value option

and region = { body : instr list; yield : value option }

type arg = { name : string; value : value }
type kernel = { name : string; args : arg list; body : instr list }
type actual = {
  (* Identifies the complete backing allocation. Distinct ids are a caller
     assertion that the allocations do not overlap; the raw CUDA ABI cannot
     verify ranges because it currently passes pointers without extents. *)
  buffer_id : int;
  actual_ownership : ownership;
  actual_permission : permission;
  actual_locality : locality;
  actual_domain_portability : domain_portability;
  actual_gpu_boundary : gpu_boundary;
}

let make_value ?(loc=Source_span.synthetic) ?(ownership=Aliased) ?(locality=Gpu_mode.Local)
    ?(domain_portability=Domain_portability_unspecified)
    ?(gpu_boundary=Boundary_unspecified) ?(permission=Read_only) ?layout
    ?(provenance=None) id ty =
  { id; ty; loc; ownership; locality; domain_portability; gpu_boundary; permission; layout; provenance }
let results = function
  | Const_bool(v,_) | Sub_i32(v,_,_) | Mul_i32(v,_,_) | Compare(v,_,_,_) -> [v]
  | If(dst,_,_,_) -> Option.to_list dst
  | Const_i32 (v, _) | Const_f32 (v, _) | Thread_idx_x v | Global_idx_x v | Gep_f32 (v, _, _) -> [v]
  | Add_i32 (v, _, _) | Add_f32 (v, _, _) | Mul_f32 (v, _, _) | Load_f32 (v, _)
  | Load_f32_masked (v, _, _, _)
  | Shared_alloc v | Load_tensor (v, _) -> [v]
  | Scale_tensor_f32 (v, _, _) -> [v]
  | Mul_tensor_f32 (v, _, _) -> [v]
  | Reduce_sum_f32 (v, _) -> [v]
  | Warp_sum_f32 (v, _) -> [v]
  | Store_f32 _ | Store_f32_masked _ | Store_f32_grid_leader _ | Store_tensor _ | Barrier _ | Return _ -> []
let operands = function
  | Const_bool _ -> []
  | Sub_i32(_,a,b) | Mul_i32(_,a,b) | Compare(_,_,a,b) -> [a;b]
  | If(_,c,_,_) -> [c]
  | Const_i32 _ | Const_f32 _ | Thread_idx_x _ | Global_idx_x _ -> []
  | Gep_f32 (_, p, ix) -> [p; ix]
  | Add_i32 (_, a, b) | Add_f32 (_, a, b) | Mul_f32 (_, a, b) -> [a; b]
  | Load_f32 (_, p) -> [p] | Load_f32_masked (_, p, ix, n) -> [p; ix; n]
  | Store_f32 (p, v) -> [p; v] | Store_f32_masked (p, v, ix, n) -> [p; v; ix; n]
  | Shared_alloc _ | Barrier _ -> []
  | Load_tensor (_, src) -> [src]
  | Scale_tensor_f32 (_, tensor, scalar) -> [tensor; scalar]
  | Mul_tensor_f32 (_, a, b) -> [a; b]
  | Reduce_sum_f32 (_, tensor) -> [tensor]
  | Warp_sum_f32 (_, value) -> [value]
  | Store_f32_grid_leader (ptr, value) -> [ptr; value]
  | Store_tensor (dst, src) -> [dst; src]
  | Return None -> [] | Return (Some v) -> [v]

(* Rebuild an instruction with transformed operand values. Definitions remain
   unchanged; nested regions and their yields are traversed as well. *)
let rec map_uses replace = function
  | Const_bool _ as i -> i
  | Sub_i32(d,a,b) -> Sub_i32(d,replace a,replace b)
  | Mul_i32(d,a,b) -> Mul_i32(d,replace a,replace b)
  | Compare(d,c,a,b) -> Compare(d,c,replace a,replace b)
  | If(d,c,a,b) ->
      let region (r : region) =
        { body=List.map (map_uses replace) r.body;
          yield=Option.map replace r.yield }
      in
      If(d,replace c,region a,region b)
  | (Const_i32 _ | Const_f32 _ | Thread_idx_x _ | Global_idx_x _ |
     Shared_alloc _ | Return None) as i -> i
  | Add_i32 (dst,a,b) -> Add_i32(dst,replace a,replace b)
  | Add_f32 (dst,a,b) -> Add_f32(dst,replace a,replace b)
  | Mul_f32 (dst,a,b) -> Mul_f32(dst,replace a,replace b)
  | Gep_f32 (dst,base,index) -> Gep_f32(dst,replace base,replace index)
  | Load_f32 (dst,ptr) -> Load_f32(dst,replace ptr)
  | Load_f32_masked (dst,ptr,index,bound) ->
      Load_f32_masked(dst,replace ptr,replace index,replace bound)
  | Store_f32 (ptr,value) -> Store_f32(replace ptr,replace value)
  | Store_f32_masked (ptr,value,index,bound) ->
      Store_f32_masked(replace ptr,replace value,replace index,replace bound)
  | Load_tensor (dst,src) -> Load_tensor(dst,replace src)
  | Store_tensor (dst,src) -> Store_tensor(replace dst,replace src)
  | Scale_tensor_f32 (dst,src,scalar) ->
      Scale_tensor_f32(dst,replace src,replace scalar)
  | Mul_tensor_f32 (dst,a,b) -> Mul_tensor_f32(dst,replace a,replace b)
  | Reduce_sum_f32 (dst,src) -> Reduce_sum_f32(dst,replace src)
  | Warp_sum_f32 (dst,src) -> Warp_sum_f32(dst,replace src)
  | Store_f32_grid_leader (ptr,value) ->
      Store_f32_grid_leader(replace ptr,replace value)
  | Barrier _ as i -> i
  | Return (Some value) -> Return (Some (replace value))

let rec flatten body = List.concat_map (fun i -> i :: (match i with
  | If(_,_,a,b) -> flatten a.body @ flatten b.body | _ -> [])) body
