open Ir

let reg v = "%r" ^ string_of_int v.id
let freg v = "%f" ^ string_of_int v.id
let ptrreg v = "%rd" ^ string_of_int v.id
let is_pointer = function
  | Gpu_type.Ptr (_, Gpu_type.Global) | Gpu_type.MemRef (_, _, Gpu_type.Global) -> true
  | _ -> false
let param_type = function
  | Gpu_type.F32 -> "f32"
  | Gpu_type.I32 -> "u32"
  | ty when is_pointer ty -> "u64"
  | ty -> invalid_arg ("PTX cannot pass argument type " ^ Gpu_type.string_of_ty ty)
let param_register value = match value.ty with
  | Gpu_type.F32 -> freg value
  | Gpu_type.I32 -> reg value
  | ty when is_pointer ty -> ptrreg value
  | ty -> invalid_arg ("PTX cannot load parameter type " ^ Gpu_type.string_of_ty ty)

let maximum_value_id kernel =
  List.fold_left (fun max_id value -> max max_id value.Ir.id) 0 (Ptx_ir.all_values kernel)

let require condition message = if not condition then invalid_arg message

let is_warp_reduction_input value =
  value.ty = Gpu_type.Tensor ([Gpu_type.Static 32], Gpu_type.Float32)
  && value.layout = Some (Layout.Register {
       elements_per_lane=[1]; lanes_per_subgroup=[32]; subgroups_per_cta=[1]; order=[0] })

(* Target legalization belongs with the PTX backend: this is a strategy choice
   for this target, not a source/semantic IR pass. *)
let lower_to_ptx_ir kernel =
  let next_id = ref (1 + List.fold_left max (-1)
      (List.map (fun arg -> arg.value.id) kernel.args @
       List.concat_map (fun instr -> List.map (fun v -> v.id) (results instr)) kernel.body)) in
  let fresh ty = let id = !next_id in incr next_id; make_value id ty in
  let body=List.concat_map (function
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
      | Mul_f32 (v,a,b) | Scale_tensor_f32 (v,a,b) | Mul_tensor_f32 (v,a,b) ->
          [Ptx_ir.Mul_f32(v,a,b)]
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
      | Return (Some _) -> invalid_arg "PTX kernel entry points cannot return a value") kernel.body in
  let uses_subgroup = List.exists (function
    | Ptx_ir.Tensor_lane_f32 _ | Ptx_ir.Warp_butterfly_sum_f32 _ -> true
    | Ptx_ir.Tile_load_f32 _ | Ptx_ir.Tile_store_f32 _ | Ptx_ir.Shared_alloc _ -> true
    | _ -> false) body in
  let shared_allocations = List.fold_left (fun ids -> function
    | Ptx_ir.Shared_alloc memref -> if List.mem memref.id ids then ids else memref.id :: ids
    | _ -> ids) [] body in
  { Ptx_ir.name=kernel.name; args=kernel.args; body;
    launch={threads_per_cta=(if uses_subgroup then Some (32,1,1) else None);
            static_shared_bytes=128 * List.length shared_allocations} }

let require_register_layout value = match value.ty, value.layout with
  | Gpu_type.Tensor ([Gpu_type.Static 32], Gpu_type.Float32),
    Some (Layout.Register { elements_per_lane=[1]; lanes_per_subgroup=[32];
                            subgroups_per_cta=[1]; order=[0] }) -> ()
  | _ -> invalid_arg "PTX tile movement supports only a 32-element f32 tensor with one element per lane in a single warp"

let require_shared_layout value = match value.ty, value.layout with
  | Gpu_type.MemRef ([Gpu_type.Static 32], Gpu_type.Float32, Gpu_type.Shared),
    Some (Layout.Shared { vector_width=1; order=[0]; swizzle=Layout.No_swizzle }) -> ()
  | _ -> invalid_arg "PTX tile movement supports only an unswizzled 32-element f32 shared layout"

let require_global_tile value = match value.ty, value.layout with
  | Gpu_type.MemRef ([Gpu_type.Static 32], Gpu_type.Float32, Gpu_type.Global), None -> ()
  | _ -> invalid_arg "PTX tile movement supports only a 32-element contiguous global f32 MemRef"

let emit kernel =
  Verifier.verify_exn kernel;
  let kernel = lower_to_ptx_ir kernel in
  let has_warp_reduction = List.exists (function
    | Ptx_ir.Warp_butterfly_sum_f32 _ -> true | _ -> false) kernel.body in
  if has_warp_reduction && List.exists (function
      | Ptx_ir.Global_idx_x _ -> true | _ -> false) kernel.body then
    invalid_arg "the current warp reduction is CTA-local and cannot consume global_idx_x values";
  let max_id = maximum_value_id kernel in
  let offset_scratch = "%rd" ^ string_of_int (max_id + 2) in
  let address_scratch = "%rd" ^ string_of_int (max_id + 3) in
  (* Values occupy ids [0..max_id]. Reserve scratch ids explicitly by role:
     tid, CTA id, CTA width, then two distinct warp-reduction temporaries. *)
  let tid_scratch_id = max_id + 1 in
  let cta_scratch_id = max_id + 2 in
  let width_scratch_id = max_id + 3 in
  let reduction_bits_scratch_id = max_id + 4 in
  let reduction_value_scratch_id = max_id + 5 in
  let register_count = max_id + 6 in
  let b32_scratch id = "%r" ^ string_of_int id in
  let f32_scratch id = "%f" ^ string_of_int id in
  let tid_scratch = b32_scratch tid_scratch_id in
  let shared_symbols = Hashtbl.create 4 in
  List.iter (function
    | Ptx_ir.Shared_alloc memref ->
        require_shared_layout memref;
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
    | Ptx_ir.Mul_f32 (dst, a, c) ->
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
    | Ptx_ir.Barrier_cta -> ()
    | Ptx_ir.Const_i32 _ | Ptx_ir.Const_f32 _ | Ptx_ir.Add_i32 _ | Ptx_ir.Add_f32 _
    | Ptx_ir.Thread_idx_x _ | Ptx_ir.Global_idx_x _ | Ptx_ir.Gep_f32 _
    | Ptx_ir.Load_f32 _ | Ptx_ir.Load_f32_masked _ | Ptx_ir.Store_f32 _
    | Ptx_ir.Store_f32_masked _ | Ptx_ir.Return -> ()) kernel.body;
  require (kernel.launch.static_shared_bytes = 128 * Hashtbl.length shared_symbols)
    "PTX shared allocation declarations disagree with the launch contract";
  let b = Buffer.create 1024 in
  let line s = Buffer.add_string b (s ^ "\n") in
  line ".version 8.5";
  line ".target sm_90";
  line ".address_size 64";
  line (".visible .entry " ^ kernel.name ^ "(");
  List.iteri (fun i arg ->
    line (Printf.sprintf "  .param .%s %%arg%d%s" (param_type arg.value.ty) i
      (if i + 1 = List.length kernel.args then "" else ","))) kernel.args;
  let launch_suffix = match kernel.launch.threads_per_cta with
    | None -> ")"
    | Some (x,y,z) -> Printf.sprintf ") .reqntid %d, %d, %d" x y z in
  line launch_suffix;
  line "{";
  Hashtbl.iter (fun _ symbol -> line ("  .shared .align 16 .b8 " ^ symbol ^ "[128];")) shared_symbols;
  line "  .reg .pred %pred<8>;";
  line (Printf.sprintf "  .reg .b32 %%r<%d>;" register_count);
  line (Printf.sprintf "  .reg .f32 %%f<%d>;" register_count);
  line (Printf.sprintf "  .reg .b64 %%rd<%d>;" register_count);
  List.iteri (fun i arg ->
    line (Printf.sprintf "  ld.param.%s %s, [%%arg%d];"
      (param_type arg.value.ty) (param_register arg.value) i)) kernel.args;
  let emit_tile_address memref =
    line (Printf.sprintf "  mov.u32 %s, %%tid.x;" tid_scratch);
    line (Printf.sprintf "  mul.wide.u32 %s, %s, 4;" offset_scratch tid_scratch);
    match memref.ty with
    | Gpu_type.MemRef (_, _, Gpu_type.Global) ->
        line (Printf.sprintf "  add.u64 %s, %s, %s;" address_scratch (ptrreg memref) offset_scratch);
        "global"
    | Gpu_type.MemRef (_, _, Gpu_type.Shared) ->
        let symbol = match Hashtbl.find_opt shared_symbols memref.id with
          | Some symbol -> symbol
          | None -> invalid_arg (Printf.sprintf "shared MemRef %d has no allocation" memref.id) in
        line (Printf.sprintf "  mov.u64 %s, %s;" address_scratch symbol);
        line (Printf.sprintf "  cvta.to.shared.u64 %s, %s;" address_scratch address_scratch);
        line (Printf.sprintf "  add.u64 %s, %s, %s;" address_scratch address_scratch offset_scratch);
        "shared"
    | _ -> invalid_arg "tile address must be global or shared MemRef" in
  let emit_core = function
    | Ptx_ir.Const_i32 (value, n) -> line (Printf.sprintf "  mov.u32 %s, %d;" (reg value) n)
    | Ptx_ir.Const_f32 (value, f) ->
        line (Printf.sprintf "  mov.f32 %s, 0f%08lx;" (freg value) (Int32.bits_of_float f))
    | Ptx_ir.Thread_idx_x value -> line (Printf.sprintf "  mov.u32 %s, %%tid.x;" (reg value))
    | Ptx_ir.Global_idx_x value ->
        line (Printf.sprintf "  mov.u32 %s, %%tid.x;" tid_scratch);
        line (Printf.sprintf "  mov.u32 %s, %%ctaid.x;" (b32_scratch cta_scratch_id));
        line (Printf.sprintf "  mov.u32 %s, %%ntid.x;" (b32_scratch width_scratch_id));
        line (Printf.sprintf "  mad.lo.u32 %s, %s, %s, %s;" (reg value)
          (b32_scratch cta_scratch_id) (b32_scratch width_scratch_id) tid_scratch)
    | Ptx_ir.Gep_f32 (dst, base, index) ->
        line (Printf.sprintf "  mul.wide.u32 %s, %s, 4;" offset_scratch (reg index));
        line (Printf.sprintf "  add.u64 %s, %s, %s;" (ptrreg dst) (ptrreg base) offset_scratch)
    | Ptx_ir.Add_i32 (dst, a, c) -> line (Printf.sprintf "  add.u32 %s, %s, %s;" (reg dst) (reg a) (reg c))
    | Ptx_ir.Add_f32 (dst, a, c) -> line (Printf.sprintf "  add.f32 %s, %s, %s;" (freg dst) (freg a) (freg c))
    | Ptx_ir.Mul_f32 (dst, a, c) -> line (Printf.sprintf "  mul.f32 %s, %s, %s;" (freg dst) (freg a) (freg c))
    | Ptx_ir.Load_f32 (dst, ptr) -> line (Printf.sprintf "  ld.global.f32 %s, [%s];" (freg dst) (ptrreg ptr))
    | Ptx_ir.Load_f32_masked (dst, ptr, ix, bound) ->
        line (Printf.sprintf "  setp.ge.s32 %%pred1, %s, 0;" (reg ix));
        line (Printf.sprintf "  setp.lt.s32 %%pred0, %s, %s;" (reg ix) (reg bound));
        line "  and.pred %pred0, %pred0, %pred1;";
        line (Printf.sprintf "  mov.f32 %s, 0f00000000;" (freg dst));
        line (Printf.sprintf "  @%%pred0 ld.global.f32 %s, [%s];" (freg dst) (ptrreg ptr))
    | Ptx_ir.Store_f32 (ptr, value) -> line (Printf.sprintf "  st.global.f32 [%s], %s;" (ptrreg ptr) (freg value))
    | Ptx_ir.Store_f32_masked (ptr, value, ix, bound) ->
        line (Printf.sprintf "  setp.ge.s32 %%pred1, %s, 0;" (reg ix));
        line (Printf.sprintf "  setp.lt.s32 %%pred0, %s, %s;" (reg ix) (reg bound));
        line "  and.pred %pred0, %pred0, %pred1;";
        line (Printf.sprintf "  @%%pred0 st.global.f32 [%s], %s;" (ptrreg ptr) (freg value))
    | Ptx_ir.Shared_alloc _ -> ()
    | Ptx_ir.Tile_load_f32 (tensor, memref) ->
        let space = emit_tile_address memref in
        line (Printf.sprintf "  ld.%s.f32 %s, [%s];" space (freg tensor) address_scratch)
    | Ptx_ir.Tile_store_f32 (memref, tensor) ->
        let space = emit_tile_address memref in
        line (Printf.sprintf "  st.%s.f32 [%s], %s;" space address_scratch (freg tensor))
    | Ptx_ir.Store_grid_leader_f32 (ptr, value) ->
        line (Printf.sprintf "  mov.u32 %s, %%tid.x;" tid_scratch);
        line (Printf.sprintf "  setp.eq.u32 %%pred0, %s, 0;" tid_scratch);
        line (Printf.sprintf "  mov.u32 %s, %%tid.y;" tid_scratch);
        line (Printf.sprintf "  setp.eq.u32 %%pred1, %s, 0;" tid_scratch);
        line "  and.pred %pred2, %pred0, %pred1;";
        line (Printf.sprintf "  mov.u32 %s, %%tid.z;" tid_scratch);
        line (Printf.sprintf "  setp.eq.u32 %%pred1, %s, 0;" tid_scratch);
        line "  and.pred %pred2, %pred2, %pred1;";
        line (Printf.sprintf "  mov.u32 %s, %%ctaid.x;" (b32_scratch cta_scratch_id));
        line (Printf.sprintf "  setp.eq.u32 %%pred1, %s, 0;" (b32_scratch cta_scratch_id));
        line "  and.pred %pred2, %pred2, %pred1;";
        line (Printf.sprintf "  mov.u32 %s, %%ctaid.y;" (b32_scratch cta_scratch_id));
        line (Printf.sprintf "  setp.eq.u32 %%pred1, %s, 0;" (b32_scratch cta_scratch_id));
        line "  and.pred %pred2, %pred2, %pred1;";
        line (Printf.sprintf "  mov.u32 %s, %%ctaid.z;" (b32_scratch cta_scratch_id));
        line (Printf.sprintf "  setp.eq.u32 %%pred1, %s, 0;" (b32_scratch cta_scratch_id));
        line "  and.pred %pred2, %pred2, %pred1;";
        line (Printf.sprintf "  @%%pred2 st.global.f32 [%s], %s;" (ptrreg ptr) (freg value))
    | Ptx_ir.Barrier_cta -> line "  bar.sync 0;"
    | Ptx_ir.Tensor_lane_f32 _ | Ptx_ir.Warp_butterfly_sum_f32 _ -> assert false
    | Ptx_ir.Return -> line "  ret;" in
  let emit_warp_sum dst src =
    let peer_bits = b32_scratch reduction_bits_scratch_id in
    let peer_value = f32_scratch reduction_value_scratch_id in
    line (Printf.sprintf "  mov.f32 %s, %s;" (freg dst) (freg src));
    List.iter (fun lane_delta ->
      line (Printf.sprintf "  mov.b32 %s, %s;" peer_bits (freg dst));
      line (Printf.sprintf "  shfl.sync.bfly.b32 %s, %s, %d, 31, 0xffffffff;" peer_bits peer_bits lane_delta);
      line (Printf.sprintf "  mov.b32 %s, %s;" peer_value peer_bits);
      line (Printf.sprintf "  add.f32 %s, %s, %s;" (freg dst) (freg dst) peer_value)) [16;8;4;2;1] in
  List.iter (function
    | Ptx_ir.Tensor_lane_f32 (dst, src) ->
        line (Printf.sprintf "  mov.f32 %s, %s;" (freg dst) (freg src))
    | Ptx_ir.Warp_butterfly_sum_f32 (dst, src) -> emit_warp_sum dst src
    | operation -> emit_core operation) kernel.body;
  line "  ret;";
  line "}";
  Buffer.contents b
