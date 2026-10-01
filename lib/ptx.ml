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

let all_values kernel =
  List.concat_map (fun instruction -> Ir.operands instruction @ Ir.results instruction) kernel.body
  @ List.map (fun arg -> arg.value) kernel.args

let maximum_value_id kernel =
  List.fold_left (fun max_id value -> max max_id value.id) 0 (all_values kernel)

let require condition message = if not condition then invalid_arg message

let is_warp_reduction_input value =
  value.ty = Gpu_type.Tensor ([Gpu_type.Static 32], Gpu_type.Float32)
  && value.layout = Some (Layout.Register {
       size_per_thread=[1]; threads_per_warp=[32]; warps_per_cta=[1]; order=[0] })

(* Target legalization belongs with the PTX backend: this is a strategy choice
   for this target, not a source/semantic IR pass. *)
let lower_reductions kernel =
  let next_id = ref (1 + List.fold_left max (-1)
      (List.map (fun arg -> arg.value.id) kernel.args @
       List.concat_map (fun instr -> List.map (fun v -> v.id) (results instr)) kernel.body)) in
  let fresh ty = let id = !next_id in incr next_id; make_value id ty in
  { kernel with body=List.concat_map (function
      | Reduce_sum_f32 (dst, src) ->
          if not (is_warp_reduction_input src) then
            invalid_arg "PTX reduction requires Tensor<[32], f32> with one element per lane in one warp";
          let lane_value = fresh Gpu_type.F32 in
          [Tensor_lane_f32 (lane_value, src); Warp_reduce_sum_f32 (dst, lane_value)]
      | instruction -> [instruction]) kernel.body }

let require_register_layout value = match value.ty, value.layout with
  | Gpu_type.Tensor ([Gpu_type.Static 32], Gpu_type.Float32),
    Some (Layout.Register { size_per_thread=[1]; threads_per_warp=[32];
                            warps_per_cta=[1]; order=[0] }) -> ()
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
  let kernel = lower_reductions kernel in
  Verifier.verify_exn kernel;
  let has_warp_reduction = List.exists (function Warp_reduce_sum_f32 _ -> true | _ -> false) kernel.body in
  let needs_one_warp_block = has_warp_reduction || List.exists (function
    | Load_tensor _ | Store_tensor _ | Scale_tensor_f32 _ | Mul_tensor_f32 _
    | Tensor_lane_f32 _ | Shared_alloc _ -> true
    | _ -> false) kernel.body in
  if has_warp_reduction && List.exists (function Global_idx_x _ -> true | _ -> false) kernel.body then
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
    | Shared_alloc memref ->
        require_shared_layout memref;
        Hashtbl.replace shared_symbols memref.id ("__shared_" ^ string_of_int memref.id)
    | Load_tensor (tensor, memref) ->
        require_register_layout tensor;
        (match memref.ty with
         | Gpu_type.MemRef (_, _, Gpu_type.Shared) -> require_shared_layout memref
         | Gpu_type.MemRef (_, _, Gpu_type.Global) -> require_global_tile memref
         | _ -> invalid_arg "PTX tile load source must be global or shared memory")
    | Store_tensor (memref, tensor) ->
        require_register_layout tensor;
        (match memref.ty with
         | Gpu_type.MemRef (_, _, Gpu_type.Shared) -> require_shared_layout memref
         | Gpu_type.MemRef (_, _, Gpu_type.Global) -> require_global_tile memref
        | _ -> invalid_arg "PTX tile store destination must be global or shared memory")
    | Scale_tensor_f32 (dst, src, _) -> require_register_layout dst; require_register_layout src
    | Mul_tensor_f32 (dst, a, c) ->
        require_register_layout dst; require_register_layout a; require_register_layout c
    | Reduce_sum_f32 _ -> invalid_arg "unlowered shape-level reduction reached PTX emission"
    | Tensor_lane_f32 (_, src) -> require_register_layout src
    | Warp_reduce_sum_f32 (_, src) -> require (src.ty = Gpu_type.F32) "warp sum requires f32 lane values"
    | Store_f32_grid_leader (ptr, value) ->
        require (is_pointer ptr.ty) "grid-leader store requires a global f32 buffer";
        require (value.ty = Gpu_type.F32) "grid-leader store value must be f32"
    | Barrier Execution.Cta -> ()
    | Barrier _ -> invalid_arg "PTX tile movement currently supports CTA barriers only"
    | _ -> ()) kernel.body;
  let b = Buffer.create 1024 in
  let line s = Buffer.add_string b (s ^ "\n") in
  line ".version 8.5";
  line ".target sm_90";
  line ".address_size 64";
  line (".visible .entry " ^ kernel.name ^ "(");
  List.iteri (fun i arg ->
    line (Printf.sprintf "  .param .%s %%arg%d%s" (param_type arg.value.ty) i
      (if i + 1 = List.length kernel.args then "" else ","))) kernel.args;
  line (if needs_one_warp_block then ") .reqntid 32, 1, 1" else ")");
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
  List.iter (function
    | Const_i32 (value, n) -> line (Printf.sprintf "  mov.u32 %s, %d;" (reg value) n)
    | Const_f32 (value, f) ->
        line (Printf.sprintf "  mov.f32 %s, 0f%08lx;" (freg value) (Int32.bits_of_float f))
    | Thread_idx_x value -> line (Printf.sprintf "  mov.u32 %s, %%tid.x;" (reg value))
    | Global_idx_x value ->
        line (Printf.sprintf "  mov.u32 %s, %%tid.x;" tid_scratch);
        line (Printf.sprintf "  mov.u32 %s, %%ctaid.x;" (b32_scratch cta_scratch_id));
        line (Printf.sprintf "  mov.u32 %s, %%ntid.x;" (b32_scratch width_scratch_id));
        line (Printf.sprintf "  mad.lo.u32 %s, %s, %s, %s;" (reg value)
          (b32_scratch cta_scratch_id) (b32_scratch width_scratch_id) tid_scratch)
    | Gep_f32 (dst, base, index) ->
        line (Printf.sprintf "  mul.wide.u32 %s, %s, 4;" offset_scratch (reg index));
        line (Printf.sprintf "  add.u64 %s, %s, %s;" (ptrreg dst) (ptrreg base) offset_scratch)
    | Add_i32 (dst, a, c) -> line (Printf.sprintf "  add.u32 %s, %s, %s;" (reg dst) (reg a) (reg c))
    | Add_f32 (dst, a, c) -> line (Printf.sprintf "  add.f32 %s, %s, %s;" (freg dst) (freg a) (freg c))
    | Mul_f32 (dst, a, c) -> line (Printf.sprintf "  mul.f32 %s, %s, %s;" (freg dst) (freg a) (freg c))
    | Load_f32 (dst, ptr) -> line (Printf.sprintf "  ld.global.f32 %s, [%s];" (freg dst) (ptrreg ptr))
    | Load_f32_masked (dst, ptr, ix, bound) ->
        line (Printf.sprintf "  setp.ge.s32 %%pred1, %s, 0;" (reg ix));
        line (Printf.sprintf "  setp.lt.s32 %%pred0, %s, %s;" (reg ix) (reg bound));
        line "  and.pred %pred0, %pred0, %pred1;";
        line (Printf.sprintf "  mov.f32 %s, 0f00000000;" (freg dst));
        line (Printf.sprintf "  @%%pred0 ld.global.f32 %s, [%s];" (freg dst) (ptrreg ptr))
    | Store_f32 (ptr, value) -> line (Printf.sprintf "  st.global.f32 [%s], %s;" (ptrreg ptr) (freg value))
    | Store_f32_masked (ptr, value, ix, bound) ->
        line (Printf.sprintf "  setp.ge.s32 %%pred1, %s, 0;" (reg ix));
        line (Printf.sprintf "  setp.lt.s32 %%pred0, %s, %s;" (reg ix) (reg bound));
        line "  and.pred %pred0, %pred0, %pred1;";
        line (Printf.sprintf "  @%%pred0 st.global.f32 [%s], %s;" (ptrreg ptr) (freg value))
    | Shared_alloc _ -> ()
    | Load_tensor (tensor, memref) ->
        let space = emit_tile_address memref in
        line (Printf.sprintf "  ld.%s.f32 %s, [%s];" space (freg tensor) address_scratch)
    | Store_tensor (memref, tensor) ->
        let space = emit_tile_address memref in
        line (Printf.sprintf "  st.%s.f32 [%s], %s;" space address_scratch (freg tensor))
    | Scale_tensor_f32 (dst, src, scalar) ->
        line (Printf.sprintf "  mul.f32 %s, %s, %s;" (freg dst) (freg src) (freg scalar))
    | Mul_tensor_f32 (dst, a, c) ->
        line (Printf.sprintf "  mul.f32 %s, %s, %s;" (freg dst) (freg a) (freg c))
    | Reduce_sum_f32 _ -> assert false
    | Tensor_lane_f32 (dst, src) ->
        line (Printf.sprintf "  mov.f32 %s, %s;" (freg dst) (freg src))
    | Warp_reduce_sum_f32 (dst, src) ->
        let peer_bits = b32_scratch reduction_bits_scratch_id in
        let peer_value = f32_scratch reduction_value_scratch_id in
        line (Printf.sprintf "  mov.f32 %s, %s;" (freg dst) (freg src));
        List.iter (fun lane_delta ->
          line (Printf.sprintf "  mov.b32 %s, %s;" peer_bits (freg dst));
          line (Printf.sprintf "  shfl.sync.bfly.b32 %s, %s, %d, 31, 0xffffffff;" peer_bits peer_bits lane_delta);
          line (Printf.sprintf "  mov.b32 %s, %s;" peer_value peer_bits);
          line (Printf.sprintf "  add.f32 %s, %s, %s;" (freg dst) (freg dst) peer_value)) [16;8;4;2;1]
    | Store_f32_grid_leader (ptr, value) ->
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
    | Barrier Execution.Cta -> line "  bar.sync 0;"
    | Barrier _ -> assert false
    | Return None -> line "  ret;"
    | Return (Some _) -> invalid_arg "PTX kernel entry points cannot return a value") kernel.body;
  line "  ret;";
  line "}";
  Buffer.contents b
