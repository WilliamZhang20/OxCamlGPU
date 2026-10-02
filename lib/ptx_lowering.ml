open Ir
let require condition message = if not condition then invalid_arg message
let add_shared_bytes total bytes =
  if total > max_int - bytes then invalid_arg "static shared-memory size overflows the host integer range";
  total + bytes
let is_pointer = Gpu_type.is_global_f32_buffer
let is_warp_reduction_input value =
  value.ty = Gpu_type.Tensor ([Gpu_type.Static 32], Gpu_type.Float32)
  && value.layout = Some (Layout.Register {
       elements_per_lane=[1]; lanes_per_subgroup=[32]; subgroups_per_cta=[1]; order=[0] })

(* Target legalization belongs with the PTX backend: this is a strategy choice
   for this target, not a source/semantic IR pass. *)
let lower_unchecked kernel =
  let next_id = ref (1 + List.fold_left max (-1)
      (List.map (fun arg -> arg.value.id) kernel.args @
       List.concat_map (fun instr -> List.map (fun v -> v.id) (results instr)) (Ir.flatten kernel.body))) in
  let fresh ty = let id = !next_id in incr next_id; make_value id ty in
  let labels=ref 0 in
  let label () = let n = !labels in incr labels;n in
  let rec lower body = List.concat_map (function
      | If(dst,c,a,b) ->
          let yes=label() and no=label() and done_=label() in
          let edge r = match dst,r.yield with Some d,Some v->[Ptx_ir.Move(d,v)] | None,None->[] | _->invalid_arg "invalid branch yield" in
          [Ptx_ir.Branch(c,yes,no);Ptx_ir.Label yes] @ lower a.body @ edge a @
          [Ptx_ir.Jump done_;Ptx_ir.Label no] @ lower b.body @ edge b @ [Ptx_ir.Label done_]
      | Const_bool(v,b) -> [Ptx_ir.Const_bool(v,b)]
      | Compare(d,c,a,b) -> [Ptx_ir.Compare(d,c,a,b)]
      | Sub_i32(d,a,b) -> [Ptx_ir.Sub_i32(d,a,b)]
      | Mul_i32(d,a,b) -> [Ptx_ir.Mul_i32(d,a,b)]
      | Reduce_sum_f32 (dst, src) ->
          if not (is_warp_reduction_input src) then
            invalid_arg "PTX reduction requires Tensor<[32], f32> with one element per lane in one warp";
          let lane_value = fresh Gpu_type.F32 in
          [Ptx_ir.Tensor_lane_f32 (lane_value, src);
           Ptx_ir.Warp_butterfly_sum_f32 (dst, lane_value)]
      | Warp_sum_f32 (dst, src) -> [Ptx_ir.Warp_butterfly_sum_f32 (dst, src)]
      | Const_i32 (v,n) -> [Ptx_ir.Const_i32(v,n)]
      | Const_f32 (v,n) -> [Ptx_ir.Const_f32(v,n)]
      | Add_i32 (v,a,b) -> [Ptx_ir.Add_i32(v,a,b)]
      | Add_f32 (v,a,b) -> [Ptx_ir.Add_f32(v,a,b)]
      | Mul_f32 (v,a,b) -> [Ptx_ir.Mul_f32(v,a,b)]
      | Scale_tensor_f32 (v,a,b) -> [Ptx_ir.Tensor_scale_f32(v,a,b)]
      | Mul_tensor_f32 (v,a,b) -> [Ptx_ir.Tensor_mul_f32(v,a,b)]
      | Thread_idx_x v -> [Ptx_ir.Thread_idx_x v]
      | Global_idx_x v -> [Ptx_ir.Global_idx_x v]
      | Gep_f32 (v,p,i) -> [Ptx_ir.Gep_f32(v,p,i)]
      | Load_f32 (v,p) -> [Ptx_ir.Load_f32(v,p)]
      | Load_f32_masked (v,p,i,n) -> [Ptx_ir.Load_f32_masked(v,p,i,n)]
      | Store_f32 (p,v) -> [Ptx_ir.Store_f32(p,v)]
      | Store_f32_masked (p,v,i,n) -> [Ptx_ir.Store_f32_masked(p,v,i,n)]
      | Shared_alloc v -> [Ptx_ir.Shared_alloc v]
      | Load_tensor (v,p) -> [Ptx_ir.Tile_load_f32(v,p)]
      | Store_tensor (p,v) -> [Ptx_ir.Tile_store_f32(p,v)]
      | Store_f32_grid_leader (p,v) -> [Ptx_ir.Store_grid_leader_f32(p,v)]
      | Barrier Execution.Cta -> [Ptx_ir.Barrier_cta]
      | Barrier _ -> invalid_arg "PTX target lowering supports CTA barriers only"
      | Return None -> [Ptx_ir.Return]
      | Return (Some _) -> invalid_arg "PTX kernel entry points cannot return a value") body in
  let body=lower kernel.body in
  let uses_subgroup = List.exists (function
    | Ptx_ir.Tensor_lane_f32 _ | Ptx_ir.Warp_butterfly_sum_f32 _ -> true
    | Ptx_ir.Tile_load_f32 _ | Ptx_ir.Tile_store_f32 _ | Ptx_ir.Shared_alloc _ -> true
    | _ -> false) body in
  let shared_bytes = List.fold_left (fun (seen, total) -> function
    | Ptx_ir.Shared_alloc memref when not (List.mem memref.id seen) ->
        (match memref.ty, memref.layout with
         | Gpu_type.MemRef (shape, dtype, Gpu_type.Shared), Some layout ->
             (memref.id :: seen, add_shared_bytes total
                (Layout.shared_storage_bytes shape dtype layout))
         | _ -> invalid_arg "shared allocation needs a statically sized shared MemRef layout")
    | _ -> seen, total) ([], 0) body |> snd in
  { Ptx_ir.name=kernel.name; args=kernel.args; body;
    (* Subgroup collectives are legal only under this one-subgroup launch
       strategy. The PTX strategy currently fixes that subgroup at 32 lanes. *)
    launch={threads_per_cta=(if uses_subgroup then Some (32,1,1) else None);
            static_shared_bytes=shared_bytes} }

let require_register_layout value = match value.ty, value.layout with
  | Gpu_type.Tensor ([Gpu_type.Static 32], Gpu_type.Float32),
    Some (Layout.Register { elements_per_lane=[1]; lanes_per_subgroup=[32];
                            subgroups_per_cta=[1]; order=[0] }) -> ()
  | Gpu_type.Tensor (_, _), Some (Layout.Register r)
    when List.fold_left ( * ) 1 r.elements_per_lane > 1 ->
      invalid_arg "PTX backend does not yet expand multi-register tensor values"
  | _ -> invalid_arg "PTX tile movement supports only a 32-element f32 tensor with one element per lane in a single warp"

let require_shared_layout value = match value.ty, value.layout with
  | Gpu_type.MemRef ([Gpu_type.Static 32], Gpu_type.Float32, Gpu_type.Shared),
    Some (Layout.Shared { vector_width=1; order=[0]; swizzle=Layout.No_swizzle }) -> ()
  | _ -> invalid_arg "PTX tile movement supports only an unswizzled 32-element f32 shared layout"

let require_global_tile value = match value.ty, value.layout with
  | Gpu_type.MemRef ([Gpu_type.Static 32], Gpu_type.Float32, Gpu_type.Global), None -> ()
  | _ -> invalid_arg "PTX tile movement supports only a 32-element contiguous global f32 MemRef"


let lower kernel =
  Verifier.verify_exn kernel;
  let kernel=lower_unchecked kernel in
  let shared_symbols = Hashtbl.create 4 in
  List.iter (function
    | Ptx_ir.Shared_alloc memref ->
        (match memref.ty, memref.layout with
         | Gpu_type.MemRef (shape, dtype, Gpu_type.Shared), Some (Layout.Shared _ as layout) ->
             (match Layout.validate shape layout with
              | Ok () -> ignore (Gpu_type.storage_bytes shape dtype)
              | Error message -> invalid_arg ("invalid shared allocation layout: " ^ message))
         | _ -> invalid_arg "shared allocation needs a statically shaped shared MemRef layout");
        Hashtbl.replace shared_symbols memref.id ("__shared_" ^ string_of_int memref.id)
    | Ptx_ir.Tile_load_f32 (tensor, memref) ->
        require_register_layout tensor;
        (match memref.ty with
         | Gpu_type.MemRef (_, _, Gpu_type.Shared) -> require_shared_layout memref
         | Gpu_type.MemRef (_, _, Gpu_type.Global) -> require_global_tile memref
         | _ -> invalid_arg "PTX tile load source must be global or shared memory")
    | Ptx_ir.Tile_store_f32 (memref, tensor) ->
        require_register_layout tensor;
        (match memref.ty with
         | Gpu_type.MemRef (_, _, Gpu_type.Shared) -> require_shared_layout memref
         | Gpu_type.MemRef (_, _, Gpu_type.Global) -> require_global_tile memref
        | _ -> invalid_arg "PTX tile store destination must be global or shared memory")
    | Ptx_ir.Mul_f32 (dst, a, c)
    | Ptx_ir.Tensor_scale_f32 (dst, a, c)
    | Ptx_ir.Tensor_mul_f32 (dst, a, c) ->
        (match dst.ty with Gpu_type.Tensor _ ->
           require_register_layout dst; require_register_layout a
         | _ -> ());
        (match c.ty with Gpu_type.Tensor _ -> require_register_layout c | _ -> ())
    | Ptx_ir.Tensor_lane_f32 (_, src) -> require_register_layout src
    | Ptx_ir.Warp_butterfly_sum_f32 (_, src) ->
        require (src.ty = Gpu_type.F32) "warp sum requires f32 lane values"
    | Ptx_ir.Store_grid_leader_f32 (ptr, value) ->
        require (is_pointer ptr.ty) "grid-leader store requires a global f32 buffer";
        require (value.ty = Gpu_type.F32) "grid-leader store value must be f32"
    | Ptx_ir.Const_bool _ | Ptx_ir.Sub_i32 _ | Ptx_ir.Mul_i32 _ | Ptx_ir.Compare _
    | Ptx_ir.Label _ | Ptx_ir.Branch _ | Ptx_ir.Jump _ | Ptx_ir.Move _
    | Ptx_ir.Barrier_cta -> ()
    | Ptx_ir.Const_i32 _ | Ptx_ir.Const_f32 _ | Ptx_ir.Add_i32 _ | Ptx_ir.Add_f32 _
    | Ptx_ir.Thread_idx_x _ | Ptx_ir.Global_idx_x _ | Ptx_ir.Gep_f32 _
    | Ptx_ir.Load_f32 _ | Ptx_ir.Load_f32_masked _ | Ptx_ir.Store_f32 _
    | Ptx_ir.Store_f32_masked _ | Ptx_ir.Return -> ()) kernel.body;
  let expected_shared_bytes = Hashtbl.fold (fun id _ total ->
    let memref = List.find (function Ptx_ir.Shared_alloc v -> v.id = id | _ -> false) kernel.body in
    match memref with
    | Ptx_ir.Shared_alloc { ty=Gpu_type.MemRef (shape, dtype, Gpu_type.Shared); layout=Some layout; _ } ->
        add_shared_bytes total (Layout.shared_storage_bytes shape dtype layout)
    | _ -> assert false)
      shared_symbols 0 in
  require (kernel.launch.static_shared_bytes = expected_shared_bytes)
    "PTX shared allocation declarations disagree with the launch contract";
  kernel
