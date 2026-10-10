open Ir

let reg v = "%r" ^ string_of_int v.id
let freg v = "%f" ^ string_of_int v.id
let ptrreg v = "%rd" ^ string_of_int v.id
let is_pointer = function
  | Gpu_type.Ptr (_, Gpu_type.Global) | Gpu_type.MemRef (_, _, Gpu_type.Global)
  | Gpu_type.Tensor_map -> true
  | _ -> false
let is_b64_reg = function
  | Gpu_type.U64 -> true
  | ty -> is_pointer ty
let param_type = function
  | Gpu_type.F32 -> "f32"
  | Gpu_type.I32 | Gpu_type.Bool -> "u32"
  | Gpu_type.U64 -> "u64"
  | ty when is_pointer ty -> "u64"
  | ty -> invalid_arg ("PTX cannot pass argument type " ^ Gpu_type.string_of_ty ty)
let param_register value = match value.ty with
  | Gpu_type.F32 -> freg value
  | Gpu_type.I32 | Gpu_type.Bool -> reg value
  | ty when is_b64_reg ty -> ptrreg value
  | ty -> invalid_arg ("PTX cannot load parameter type " ^ Gpu_type.string_of_ty ty)

let maximum_value_id kernel =
  List.fold_left (fun max_id value -> max max_id value.Ir.id) 0 (Ptx_ir.all_values kernel)

let emit_target (kernel : Ptx_ir.kernel) =
  List.iter (function
    | Ptx_ir.Mul_f32 (dst, a, c)
      when dst.Ir.ty <> Gpu_type.F32 || a.Ir.ty <> Gpu_type.F32 || c.Ir.ty <> Gpu_type.F32 ->
        invalid_arg "PTX scalar Mul_f32 requires scalar f32 operands; tensor operations use tensor-specific IR"
    | Ptx_ir.Tile_load_f32 (tensor, _) | Ptx_ir.Tile_store_f32 (_, tensor)
    | Ptx_ir.Tile_load_f32_masked (tensor, _, _, _) | Ptx_ir.Tile_store_f32_masked (_, tensor, _, _)
      when (match tensor.Ir.ty with Gpu_type.Tensor _ -> true | _ -> false) ->
        invalid_arg "PTX emission requires tensor tiles to pass through physicalization"
    | Ptx_ir.Tensor_scale_f32 _ | Ptx_ir.Tensor_mul_f32 _ | Ptx_ir.Tensor_lane_f32 _ ->
        invalid_arg "PTX emission received an unphysicalized tensor operation"
    | _ -> ()) kernel.body;
  let max_id = maximum_value_id kernel in
  let offset_scratch = "%rd" ^ string_of_int (max_id + 2) in
  let address_scratch = "%rd" ^ string_of_int (max_id + 3) in
  let mbar_scratch = "%rd" ^ string_of_int (max_id + 4) in
  (* Values occupy ids [0..max_id]. Reserve scratch ids explicitly by role:
     tid, CTA id, CTA width, warp-reduction temps, then 2D row/col temps. *)
  let tid_scratch_id = max_id + 1 in
  let cta_scratch_id = max_id + 2 in
  let width_scratch_id = max_id + 3 in
  let reduction_bits_scratch_id = max_id + 4 in
  let reduction_value_scratch_id = max_id + 5 in
  let row_scratch_id = max_id + 6 in
  let col_scratch_id = max_id + 7 in
  let register_count = max_id + 8 in
  let wait_label = ref 0 in
  let fresh_wait_label () =
    let n = !wait_label in incr wait_label; Printf.sprintf "MBAR_WAIT_%d" n
  in
  let b32_scratch id = "%r" ^ string_of_int id in
  let f32_scratch id = "%f" ^ string_of_int id in
  let tid_scratch = b32_scratch tid_scratch_id in
  let row_scratch = b32_scratch row_scratch_id in
  let col_scratch = b32_scratch col_scratch_id in
  let shared = Ptx_ir.shared_plan kernel.Ptx_ir.name kernel.body in
  let b = Buffer.create 1024 in
  let line s = Buffer.add_string b (s ^ "\n") in
  line ".version 8.5";
  let has_wgmma =
    List.exists (function
      | Ptx_ir.Wgmma_fence | Ptx_ir.Wgmma_mma_tf32 _
      | Ptx_ir.Gmma_descriptor _
      | Ptx_ir.Wgmma_commit_group | Ptx_ir.Wgmma_wait_group _ -> true
      | _ -> false) kernel.body
  in
  line (if has_wgmma then ".target sm_90a" else ".target sm_90");
  line ".address_size 64";
  (* A launch needs these numbers; keep them in the artifact, not just the API. *)
  line (Printf.sprintf "// oxgpu.shared.total %d" shared.Ptx_ir.total_bytes);
  line (Printf.sprintf "// oxgpu.shared.dynamic %d" shared.Ptx_ir.dynamic_bytes);
  (* One window for every shared allocation, addressed by offset. Over the
     static cap it must be the dynamic window, which the launch sizes. Both
     forms are module scope: ptxas accepts .extern .shared nowhere else. *)
  if shared.Ptx_ir.total_bytes > 0 then
    line
      (if shared.Ptx_ir.dynamic_bytes > 0 then
         Printf.sprintf ".extern .shared .align %d .b8 %s[];"
           Ptx_ir.shared_window_align Ptx_ir.shared_window_symbol
       else
         Printf.sprintf ".shared .align %d .b8 %s[%d];"
           Ptx_ir.shared_window_align Ptx_ir.shared_window_symbol
           shared.Ptx_ir.total_bytes);
  line (".visible .entry " ^ kernel.name ^ "(");
  List.iteri (fun i arg ->
    line (Printf.sprintf "  .param .%s %%arg%d%s" (param_type arg.value.ty) i
      (if i + 1 = List.length kernel.args then "" else ","))) kernel.args;
  let launch_suffix = match kernel.launch.threads_per_cta with
    | None -> ")"
    | Some (x,y,z) -> Printf.sprintf ") .reqntid %d, %d, %d" x y z in
  line launch_suffix;
  line "{";
  line "  .reg .pred %pred<8>;";
  line (Printf.sprintf "  .reg .pred %%p<%d>;" register_count);
  line (Printf.sprintf "  .reg .b32 %%r<%d>;" register_count);
  line (Printf.sprintf "  .reg .f32 %%f<%d>;" register_count);
  line (Printf.sprintf "  .reg .b64 %%rd<%d>;" register_count);
  List.iteri (fun i arg ->
    line (Printf.sprintf "  ld.param.%s %s, [%%arg%d];"
      (param_type arg.value.ty) (param_register arg.value) i);
    if arg.value.ty = Gpu_type.Bool then
      line (Printf.sprintf "  setp.ne.u32 %%p%d, %s, 0;" arg.value.id (reg arg.value))) kernel.args;
  (* An mbarrier slot is not an allocation: it names its set's storage at a
     dynamic index. Record set and index when the op is seen, then resolve the
     address wherever the slot is used as a base. *)
  let mbarrier_slots : (int, Ir.value * Ir.value) Hashtbl.t = Hashtbl.create 4 in
  (* Base address of one shared allocation: the window, plus its offset. *)
  let rec emit_shared_window dst_reg memref =
    match Hashtbl.find_opt mbarrier_slots memref.Ir.id with
    | Some (set, index) ->
        emit_shared_window dst_reg set;
        (* Each mbarrier is 8 bytes. *)
        line (Printf.sprintf "  mul.wide.u32 %s, %s, 8;" offset_scratch (reg index));
        line (Printf.sprintf "  add.u64 %s, %s, %s;" dst_reg dst_reg offset_scratch)
    | None ->
    let offset =
      match List.assoc_opt memref.Ir.id shared.Ptx_ir.offsets with
      | Some offset -> offset
      | None ->
          invalid_arg
            (Printf.sprintf "shared MemRef %d has no allocation" memref.Ir.id)
    in
    line (Printf.sprintf "  mov.u64 %s, %s;" dst_reg Ptx_ir.shared_window_symbol);
    line (Printf.sprintf "  cvta.to.shared.u64 %s, %s;" dst_reg dst_reg);
    if offset <> 0 then
      line (Printf.sprintf "  add.u64 %s, %s, %d;" dst_reg dst_reg offset)
  in
  let emit_tile_address memref ~base_offset ~lane_stride =
    line (Printf.sprintf "  mov.u32 %s, %%tid.x;" tid_scratch);
    line (Printf.sprintf "  mul.wide.u32 %s, %s, %d;" offset_scratch tid_scratch (4 * lane_stride));
    match memref.ty with
    | Gpu_type.MemRef (_, _, Gpu_type.Global) ->
        line (Printf.sprintf "  add.u64 %s, %s, %s;" address_scratch (ptrreg memref) offset_scratch);
        if base_offset <> 0 then line (Printf.sprintf "  add.u64 %s, %s, %d;" address_scratch address_scratch (4 * base_offset));
        "global"
    | Gpu_type.MemRef (_, _, Gpu_type.Shared) ->
        emit_shared_window address_scratch memref;
        line (Printf.sprintf "  add.u64 %s, %s, %s;" address_scratch address_scratch offset_scratch);
        if base_offset <> 0 then line (Printf.sprintf "  add.u64 %s, %s, %d;" address_scratch address_scratch (4 * base_offset));
        "shared"
    | _ -> invalid_arg "tile address must be global or shared MemRef" in
  let preg v = "%p" ^ string_of_int v.id in
  (* Shared element addresses stay linear. Swizzle_*B on a Shared layout is a
     hardware TMA/GMMA mode bit, not a software XOR of the index. *)
  let emit_shared_addr memref index_reg =
    line (Printf.sprintf "  mul.wide.u32 %s, %s, 4;" offset_scratch index_reg);
    emit_shared_window address_scratch memref;
    line (Printf.sprintf "  add.u64 %s, %s, %s;" address_scratch address_scratch offset_scratch)
  in
  let emit_shared_base dst_reg memref = emit_shared_window dst_reg memref
  in
  let pred_prefix = function
    | Some p -> Printf.sprintf "  @%s " (preg p)
    | None -> "  "
  in
  let emit_mbarrier_wait mbar parity =
    emit_shared_base address_scratch mbar;
    let lbl = fresh_wait_label () in
    line (Printf.sprintf "%s:" lbl);
    line (Printf.sprintf "  mbarrier.try_wait.parity.shared::cta.b64 %%pred0, [%s], %s;"
      address_scratch (reg parity));
    line (Printf.sprintf "  @!%%pred0 bra.uni %s;" lbl)
  in
  let emit_gmma_descriptor dst smem index fields =
    (* Pack start_address>>4 | leading<<16 | stride<<32 | layout<<62. *)
    emit_shared_addr smem (reg index);
    line (Printf.sprintf "  and.b64 %s, %s, 0x3FFFF;" (ptrreg dst) address_scratch);
    line (Printf.sprintf "  shr.u64 %s, %s, 4;" (ptrreg dst) (ptrreg dst));
    if fields.Ir.leading <> 0 then
      line (Printf.sprintf "  or.b64 %s, %s, 0x%Lx;"
        (ptrreg dst) (ptrreg dst)
        (Int64.shift_left (Int64.of_int fields.leading) 16));
    if fields.stride <> 0 then
      line (Printf.sprintf "  or.b64 %s, %s, 0x%Lx;"
        (ptrreg dst) (ptrreg dst)
        (Int64.shift_left (Int64.of_int fields.stride) 32));
    if fields.layout_type <> 0 then
      line (Printf.sprintf "  or.b64 %s, %s, 0x%Lx;"
        (ptrreg dst) (ptrreg dst)
        (Int64.shift_left (Int64.of_int fields.layout_type) 62))
  in
  let emit_bounds_pred bounds =
    match bounds with
    | None -> line "  setp.eq.u32 %pred0, 0, 0;"
    | Some (rows, cols) ->
        line (Printf.sprintf "  setp.ge.s32 %%pred0, %s, 0;" row_scratch);
        line (Printf.sprintf "  setp.lt.s32 %%pred1, %s, %s;" row_scratch (reg rows));
        line "  and.pred %pred0, %pred0, %pred1;";
        line (Printf.sprintf "  setp.ge.s32 %%pred1, %s, 0;" col_scratch);
        line "  and.pred %pred0, %pred0, %pred1;";
        line (Printf.sprintf "  setp.lt.s32 %%pred1, %s, %s;" col_scratch (reg cols));
        line "  and.pred %pred0, %pred0, %pred1;"
  in
  let emit_tid_row_col ~tile_cols ~row_offset ~col_offset =
    line (Printf.sprintf "  mov.u32 %s, %%tid.x;" tid_scratch);
    line (Printf.sprintf "  div.u32 %s, %s, %d;" row_scratch tid_scratch tile_cols);
    line (Printf.sprintf "  rem.u32 %s, %s, %d;" col_scratch tid_scratch tile_cols);
    if row_offset <> 0 then
      line (Printf.sprintf "  add.s32 %s, %s, %d;" row_scratch row_scratch row_offset);
    if col_offset <> 0 then
      line (Printf.sprintf "  add.s32 %s, %s, %d;" col_scratch col_scratch col_offset)
  in
  let emit_tile_2d_addr ~tile_cols ~row_offset ~col_offset ~row_stride ~bounds =
    emit_tid_row_col ~tile_cols ~row_offset ~col_offset;
    emit_bounds_pred bounds;
    line (Printf.sprintf "  mad.lo.s32 %s, %s, %d, %s;" tid_scratch row_scratch row_stride col_scratch);
    line (Printf.sprintf "  mul.wide.s32 %s, %s, 4;" offset_scratch tid_scratch)
  in
  let emit_tile_2d_load tensor memref tile_cols row_offset col_offset row_stride bounds =
    emit_tile_2d_addr ~tile_cols ~row_offset ~col_offset ~row_stride ~bounds;
    match memref.ty with
    | Gpu_type.MemRef (_, _, Gpu_type.Global) ->
        line (Printf.sprintf "  add.u64 %s, %s, %s;" address_scratch (ptrreg memref) offset_scratch);
        line (Printf.sprintf "  mov.f32 %s, 0f00000000;" (freg tensor));
        line (Printf.sprintf "  @%%pred0 ld.global.f32 %s, [%s];" (freg tensor) address_scratch)
    | Gpu_type.MemRef (_, _, Gpu_type.Shared) ->
        emit_shared_window address_scratch memref;
        line (Printf.sprintf "  add.u64 %s, %s, %s;" address_scratch address_scratch offset_scratch);
        line (Printf.sprintf "  mov.f32 %s, 0f00000000;" (freg tensor));
        line (Printf.sprintf "  @%%pred0 ld.shared.f32 %s, [%s];" (freg tensor) address_scratch)
    | _ -> invalid_arg "2D tile load requires global or shared MemRef"
  in
  let emit_tile_2d_store memref tensor tile_cols row_offset col_offset row_stride bounds =
    emit_tile_2d_addr ~tile_cols ~row_offset ~col_offset ~row_stride ~bounds;
    match memref.ty with
    | Gpu_type.MemRef (_, _, Gpu_type.Global) ->
        line (Printf.sprintf "  add.u64 %s, %s, %s;" address_scratch (ptrreg memref) offset_scratch);
        line (Printf.sprintf "  @%%pred0 st.global.f32 [%s], %s;" address_scratch (freg tensor))
    | Gpu_type.MemRef (_, _, Gpu_type.Shared) ->
        emit_shared_window address_scratch memref;
        line (Printf.sprintf "  add.u64 %s, %s, %s;" address_scratch address_scratch offset_scratch);
        line (Printf.sprintf "  @%%pred0 st.shared.f32 [%s], %s;" address_scratch (freg tensor))
    | _ -> invalid_arg "2D tile store requires global or shared MemRef"
  in
  let emit_grid_leader ptr value =
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
  in
  let emit_core = function
    (* --- constants / scalar arith / compare --- *)
    | Ptx_ir.Const_bool(v,b) -> line (Printf.sprintf "  setp.eq.u32 %s, 0, %d;" (preg v) (if b then 0 else 1))
    | Ptx_ir.Sub_i32(d,a,b) -> line (Printf.sprintf "  sub.u32 %s, %s, %s;" (reg d) (reg a) (reg b))
    | Ptx_ir.Mul_i32(d,a,b) -> line (Printf.sprintf "  mul.lo.u32 %s, %s, %s;" (reg d) (reg a) (reg b))
    | Ptx_ir.Div_i32(d,a,b) -> line (Printf.sprintf "  div.u32 %s, %s, %s;" (reg d) (reg a) (reg b))
    | Ptx_ir.Rem_i32(d,a,b) -> line (Printf.sprintf "  rem.u32 %s, %s, %s;" (reg d) (reg a) (reg b))
    | Ptx_ir.Compare(d,c,a,b) ->
        let cmp = Gpu_type.string_of_comparison c in
        (match a.ty with
         | Gpu_type.Bool -> line (Printf.sprintf "  xor.pred %s, %s, %s;" (preg d) (preg a) (preg b));
             if c = Gpu_type.Eq then line (Printf.sprintf "  not.pred %s, %s;" (preg d) (preg d))
         | _ ->
             let suffix, operand = if a.ty = Gpu_type.F32 then "f32", freg else "s32", reg in
             let cmp = if a.ty = Gpu_type.F32 && c = Gpu_type.Ne then "neu" else cmp in
             line (Printf.sprintf "  setp.%s.%s %s, %s, %s;" cmp suffix (preg d) (operand a) (operand b)))
    (* --- control flow --- *)
    | Ptx_ir.Label l -> line (Printf.sprintf "L%d:" l)
    | Ptx_ir.Jump l -> line (Printf.sprintf "  bra L%d;" l)
    | Ptx_ir.Jump_uni l -> line (Printf.sprintf "  bra.uni L%d;" l)
    | Ptx_ir.Branch(c,a,b) -> line (Printf.sprintf "  @%s bra L%d;" (preg c) a); line (Printf.sprintf "  bra L%d;" b)
    | Ptx_ir.Branch_uni(c,a,b) ->
        line (Printf.sprintf "  @%s bra.uni L%d;" (preg c) a);
        line (Printf.sprintf "  bra.uni L%d;" b)
    | Ptx_ir.Move(d,v) ->
        let suffix, operand =
          match d.ty with
          | Gpu_type.F32 -> "f32", freg
          | Gpu_type.Bool -> "pred", preg
          | _ -> "b32", reg
        in
        line (Printf.sprintf "  mov.%s %s, %s;" suffix (operand d) (operand v))
    | Ptx_ir.Const_i32 (value, n) -> line (Printf.sprintf "  mov.u32 %s, %d;" (reg value) n)
    | Ptx_ir.Const_f32 (value, f) ->
        line (Printf.sprintf "  mov.f32 %s, 0f%08lx;" (freg value) (Int32.bits_of_float f))
    (* --- indices --- *)
    | Ptx_ir.Thread_idx_x value -> line (Printf.sprintf "  mov.u32 %s, %%tid.x;" (reg value))
    | Ptx_ir.Block_idx_x value -> line (Printf.sprintf "  mov.u32 %s, %%ctaid.x;" (reg value))
    | Ptx_ir.Block_idx_y value -> line (Printf.sprintf "  mov.u32 %s, %%ctaid.y;" (reg value))
    | Ptx_ir.Global_idx_x value ->
        line (Printf.sprintf "  mov.u32 %s, %%tid.x;" tid_scratch);
        line (Printf.sprintf "  mov.u32 %s, %%ctaid.x;" (b32_scratch cta_scratch_id));
        line (Printf.sprintf "  mov.u32 %s, %%ntid.x;" (b32_scratch width_scratch_id));
        line (Printf.sprintf "  mad.lo.u32 %s, %s, %s, %s;" (reg value)
          (b32_scratch cta_scratch_id) (b32_scratch width_scratch_id) tid_scratch)
    (* --- addressing / scalar memory --- *)
    | Ptx_ir.Gep_f32 (dst, base, index) ->
        line (Printf.sprintf "  mul.wide.u32 %s, %s, 4;" offset_scratch (reg index));
        line (Printf.sprintf "  add.u64 %s, %s, %s;" (ptrreg dst) (ptrreg base) offset_scratch)
    | Ptx_ir.Add_i32 (dst, a, c) -> line (Printf.sprintf "  add.u32 %s, %s, %s;" (reg dst) (reg a) (reg c))
    | Ptx_ir.Add_f32 (dst, a, c) -> line (Printf.sprintf "  add.rn.f32 %s, %s, %s;" (freg dst) (freg a) (freg c))
    | Ptx_ir.Mul_f32 (dst, a, c)
    | Ptx_ir.Tensor_scale_f32 (dst, a, c)
    | Ptx_ir.Tensor_mul_f32 (dst, a, c) -> line (Printf.sprintf "  mul.rn.f32 %s, %s, %s;" (freg dst) (freg a) (freg c))
    | Ptx_ir.Mad_f32 (dst, a, b, c) ->
        line (Printf.sprintf "  fma.rn.f32 %s, %s, %s, %s;" (freg dst) (freg a) (freg b) (freg c))
    | Ptx_ir.Load_f32 (dst, ptr) -> line (Printf.sprintf "  ld.global.f32 %s, [%s];" (freg dst) (ptrreg ptr))
    | Ptx_ir.Load_f32x4 (a, b, c, d, ptr) ->
        line (Printf.sprintf "  ld.global.nc.v4.f32 {%s, %s, %s, %s}, [%s];"
          (freg a) (freg b) (freg c) (freg d) (ptrreg ptr))
    | Ptx_ir.Load_f32_masked (dst, ptr, ix, bound) ->
        line (Printf.sprintf "  setp.ge.s32 %%pred1, %s, 0;" (reg ix));
        line (Printf.sprintf "  setp.lt.s32 %%pred0, %s, %s;" (reg ix) (reg bound));
        line "  and.pred %pred0, %pred0, %pred1;";
        line (Printf.sprintf "  mov.f32 %s, 0f00000000;" (freg dst));
        line (Printf.sprintf "  @%%pred0 ld.global.f32 %s, [%s];" (freg dst) (ptrreg ptr))
    | Ptx_ir.Store_f32 (ptr, value) -> line (Printf.sprintf "  st.global.f32 [%s], %s;" (ptrreg ptr) (freg value))
    | Ptx_ir.Store_f32x2 (ptr, a, b) ->
        line (Printf.sprintf "  st.global.v2.f32 [%s], {%s, %s};"
          (ptrreg ptr) (freg a) (freg b))
    | Ptx_ir.Store_f32x4 (ptr, a, b, c, d) ->
        line (Printf.sprintf "  st.global.v4.f32 [%s], {%s, %s, %s, %s};"
          (ptrreg ptr) (freg a) (freg b) (freg c) (freg d))
    | Ptx_ir.Store_f32_masked (ptr, value, ix, bound) ->
        line (Printf.sprintf "  setp.ge.s32 %%pred1, %s, 0;" (reg ix));
        line (Printf.sprintf "  setp.lt.s32 %%pred0, %s, %s;" (reg ix) (reg bound));
        line "  and.pred %pred0, %pred0, %pred1;";
        line (Printf.sprintf "  @%%pred0 st.global.f32 [%s], %s;" (ptrreg ptr) (freg value))
    (* --- shared memory --- *)
    | Ptx_ir.Shared_alloc _ -> ()
    | Ptx_ir.Shared_load_f32 (dst, memref, index) ->
        emit_shared_addr memref (reg index);
        line (Printf.sprintf "  ld.shared.f32 %s, [%s];" (freg dst) address_scratch)
    | Ptx_ir.Shared_load_f32x4 (a, b, c, d, memref, index) ->
        emit_shared_addr memref (reg index);
        line (Printf.sprintf "  ld.shared.v4.f32 {%s, %s, %s, %s}, [%s];"
          (freg a) (freg b) (freg c) (freg d) address_scratch)
    | Ptx_ir.Shared_store_f32 (memref, value, index) ->
        emit_shared_addr memref (reg index);
        line (Printf.sprintf "  st.shared.f32 [%s], %s;" address_scratch (freg value))
    | Ptx_ir.Shared_store_f32x4 (memref, a, b, c, d, index) ->
        emit_shared_addr memref (reg index);
        line (Printf.sprintf "  st.shared.v4.f32 [%s], {%s, %s, %s, %s};"
          address_scratch (freg a) (freg b) (freg c) (freg d))
    (* --- async copy / mbarrier / TMA / WGMMA --- *)
    | Ptx_ir.Cp_async_shared_f32x4 (memref, index, ptr) ->
        emit_shared_addr memref (reg index);
        (* `.cg` — cache at global level; better for streaming GEMM tiles than `.ca`. *)
        line (Printf.sprintf "  cp.async.cg.shared.global [%s], [%s], 16;"
          address_scratch (ptrreg ptr))
    | Ptx_ir.Cp_async_commit -> line "  cp.async.commit_group;"
    | Ptx_ir.Cp_async_wait_all -> line "  cp.async.wait_group 0;"
    | Ptx_ir.Mbarrier_init (mbar, count, pred) ->
        emit_shared_base address_scratch mbar;
        line (Printf.sprintf "%smbarrier.init.shared::cta.b64 [%s], %s;"
          (pred_prefix pred) address_scratch (reg count))
    | Ptx_ir.Mbarrier_arrive_expect_tx (mbar, tx, pred) ->
        emit_shared_base address_scratch mbar;
        line (Printf.sprintf "%smbarrier.arrive.expect_tx.shared::cta.b64 %s, [%s], %s;"
          (pred_prefix pred) mbar_scratch address_scratch (reg tx))
    | Ptx_ir.Mbarrier_arrive (mbar, pred) ->
        emit_shared_base address_scratch mbar;
        line (Printf.sprintf "%smbarrier.arrive.shared::cta.b64 %s, [%s];"
          (pred_prefix pred) mbar_scratch address_scratch)
    | Ptx_ir.Mbarrier_slot (dst, set, index) ->
        Hashtbl.replace mbarrier_slots dst.Ir.id (set, index)
    | Ptx_ir.Mbarrier_try_wait_parity (mbar, parity) ->
        emit_mbarrier_wait mbar parity
    | Ptx_ir.Tma_load_2d (smem, index, tmap, c0, c1, mbar, pred) ->
        emit_shared_addr smem (reg index);
        emit_shared_base mbar_scratch mbar;
        line (Printf.sprintf
          "%scp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%s], [%s, {%s, %s}], [%s];"
          (pred_prefix pred) address_scratch (ptrreg tmap) (reg c0) (reg c1) mbar_scratch)
    | Ptx_ir.Fence_proxy_async ->
        line "  fence.proxy.async.shared::cta;"
    | Ptx_ir.Wgmma_fence ->
        line "  wgmma.fence.sync.aligned;"
    | Ptx_ir.Wgmma_commit_group ->
        line "  wgmma.commit_group.sync.aligned;"
    | Ptx_ir.Wgmma_wait_group n ->
        line (Printf.sprintf "  wgmma.wait_group.sync.aligned %d;" n)
    | Ptx_ir.Gmma_descriptor (dst, smem, index, fields) ->
        emit_gmma_descriptor dst smem index fields
    | Ptx_ir.Wgmma_mma_tf32 (acc, desc_a, desc_b, n, scale_d) ->
        let acc_regs = String.concat ", " (List.map freg acc) in
        let scale = if scale_d then 1 else 0 in
        line (Printf.sprintf
          "  wgmma.mma_async.sync.aligned.m64n%dk8.f32.tf32.tf32 {%s}, %s, %s, %d, 1, 1;"
          n acc_regs (ptrreg desc_a) (ptrreg desc_b) scale)
    (* --- physicalized tiles --- *)
    | Ptx_ir.Tile_load_f32 (tensor, memref) ->
        let space = emit_tile_address memref ~base_offset:0 ~lane_stride:1 in
        line (Printf.sprintf "  ld.%s.f32 %s, [%s];" space (freg tensor) address_scratch)
    | Ptx_ir.Tile_load_f32_indexed (tensor, memref, base_offset, lane_stride) ->
        let space = emit_tile_address memref ~base_offset ~lane_stride in
        line (Printf.sprintf "  ld.%s.f32 %s, [%s];" space (freg tensor) address_scratch)
    | Ptx_ir.Tile_store_f32 (memref, tensor) ->
        let space = emit_tile_address memref ~base_offset:0 ~lane_stride:1 in
        line (Printf.sprintf "  st.%s.f32 [%s], %s;" space address_scratch (freg tensor))
    | Ptx_ir.Tile_store_f32_indexed (memref, tensor, base_offset, lane_stride) ->
        let space = emit_tile_address memref ~base_offset ~lane_stride in
        line (Printf.sprintf "  st.%s.f32 [%s], %s;" space address_scratch (freg tensor))
    | Ptx_ir.Tile_load_f32_masked _ | Ptx_ir.Tile_store_f32_masked _ ->
        invalid_arg "PTX emission requires masked tiles to pass through physicalization"
    | Ptx_ir.Tile_load_f32_2d (tensor, memref, tile_cols, row_offset, col_offset, row_stride, bounds) ->
        emit_tile_2d_load tensor memref tile_cols row_offset col_offset row_stride bounds
    | Ptx_ir.Tile_store_f32_2d (memref, tensor, tile_cols, row_offset, col_offset, row_stride, bounds) ->
        emit_tile_2d_store memref tensor tile_cols row_offset col_offset row_stride bounds
    | Ptx_ir.Store_grid_leader_f32 (ptr, value) ->
        emit_grid_leader ptr value
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
      line (Printf.sprintf "  add.rn.f32 %s, %s, %s;" (freg dst) (freg dst) peer_value)) [16;8;4;2;1] in
  List.iter (function
    | Ptx_ir.Tensor_lane_f32 (dst, src) ->
        line (Printf.sprintf "  mov.f32 %s, %s;" (freg dst) (freg src))
    | Ptx_ir.Warp_butterfly_sum_f32 (dst, src) -> emit_warp_sum dst src
    | operation -> emit_core operation) kernel.body;
  line "  ret;";
  line "}";
  Buffer.contents b

let emit_physical (kernel : Physical_ir.kernel) =
  (* Keep the physical map at the emission boundary so a future multi-register
     tensor expansion changes this stage without changing semantic SSA. *)
  List.iter (fun operation ->
    List.iter (fun value ->
      match value.Ir.ty with
      | Gpu_type.Tensor _ when Physical_ir.lookup kernel value = None ->
          invalid_arg "PTX emission received a tensor without a physical register mapping"
      | _ -> ()) (Ptx_ir.operands operation @ Ptx_ir.results operation)) kernel.target.Ptx_ir.body;
  emit_target kernel.target

let emit kernel = emit_physical (Physical_ir.physicalize (Ptx_lowering.lower kernel))

type shared_requirement = { total_bytes : int; dynamic_bytes : int }

let shared_requirement (kernel : Physical_ir.kernel) =
  let plan =
    Ptx_ir.shared_plan kernel.target.Ptx_ir.name kernel.target.Ptx_ir.body in
  { total_bytes = plan.Ptx_ir.total_bytes;
    dynamic_bytes = plan.Ptx_ir.dynamic_bytes }
