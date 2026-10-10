open Gpu_type
open Execution
open Gpu_mode
open Ir

let expect_ok = function Ok () -> () | Error es -> failwith (String.concat "; " (List.map (fun e -> e.Verifier.code ^ ": " ^ e.message) es))
let expect_error code = function
  | Error es when List.exists (fun e -> e.Verifier.code = code) es -> ()
  | Error es -> failwith ("expected " ^ code ^ ", got " ^ String.concat "," (List.map (fun e -> e.Verifier.code) es))
  | Ok () -> failwith ("expected error " ^ code)
let actual ?(ownership=Aliased) ?(permission=Read_only) ?(locality=Global)
    ?(domain_portability=Domain_portability_unspecified)
    ?(gpu_boundary=Boundary_unspecified) buffer_id =
  { buffer_id; actual_ownership=ownership; actual_permission=permission;
    actual_locality=locality; actual_domain_portability=domain_portability;
    actual_gpu_boundary=gpu_boundary }
let contains text needle =
  let n=String.length needle in
  let rec loop i = i + n <= String.length text &&
    (String.sub text i n = needle || loop (i+1)) in loop 0
let count text needle =
  let n=String.length needle in
  let rec loop i count =
    if i + n > String.length text then count
    else if String.sub text i n = needle then loop (i+n) (count+1)
    else loop (i+1) count in
  loop 0 0

let () =
  let x=make_value ~locality:Global ~permission:Read_only 0 (Ptr(F32,Gpu_type.Global))
  and y=make_value ~ownership:Unique ~locality:Global ~permission:Read_write 1 (Ptr(F32,Gpu_type.Global)) in
  let kernel={name="alias_check";args=[{name="x";value=x};{name="y";value=y}];body=[];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel kernel);
  expect_ok (Verifier.check_call kernel [actual 10;actual ~ownership:Unique ~permission:Read_write 11]);
  expect_error "E_UNIQUE_ALIAS" (Verifier.check_call kernel [actual 10;actual ~ownership:Unique ~permission:Read_write 10]);
  expect_error "E_UNIQUE_REQUIRED" (Verifier.check_call kernel [actual 10;actual ~permission:Read_write 11]);
  let modal_arg=make_value ~locality:Global ~domain_portability:Domain_portable
      ~gpu_boundary:Boundary_portable 105 (Ptr(F32,Gpu_type.Global)) in
  let modal_kernel={name="modal_call";args=[{name="p";value=modal_arg}];body=[];threads_per_cta=None} in
  expect_error "E_LOCALITY_REQUIRED"
    (Verifier.check_call modal_kernel [actual ~locality:Local 105]);
  expect_error "E_DOMAIN_PORTABILITY_REQUIRED"
    (Verifier.check_call modal_kernel [actual ~domain_portability:Domain_nonportable 105]);
  expect_error "E_DOMAIN_PORTABILITY_REQUIRED"
    (Verifier.check_call modal_kernel [actual 105]);
  expect_error "E_GPU_BOUNDARY_REQUIRED"
    (Verifier.check_call modal_kernel [actual ~gpu_boundary:Boundary_local 105]);
  expect_error "E_GPU_BOUNDARY_REQUIRED"
    (Verifier.check_call modal_kernel [actual 105]);
  expect_ok (Verifier.check_call modal_kernel [actual ~domain_portability:Domain_portable
      ~gpu_boundary:Boundary_portable 105]);
  let nonunit : Kernel_ast.t = {name="nonunit";args=[];body=[];threads=None;
    result={loc=Source_span.synthetic;ty=F32;ownership=Aliased;locality=Global;
      domain_portability=Domain_nonportable;gpu_boundary=Boundary_unspecified;permission=Read_only}} in
  (try ignore(Kernel_frontend.lower nonunit);failwith "nonunit kernel accepted"
   with Invalid_argument _ -> ());
  let local = make_value ~locality:Local 40 (Ptr(F32,Gpu_type.Global)) in
  let returning = { kernel with name="bad_escape"; body=[Return(Some local)] } in
  expect_error "E_KERNEL_RETURN_TYPE" (Verifier.verify_kernel returning);
  let ro=make_value ~permission:Read_only 50 (Ptr(F32,Gpu_type.Global)) and v=make_value 51 F32 in
  let readonly_store={name="bad_store";args=[{name="p";value=ro}];body=[Const_f32(v,1.);Store_f32(ro,v)];threads_per_cta=None} in
  expect_error "E_READONLY_STORE" (Verifier.verify_kernel readonly_store);
  let writable_ptr = { ro with permission=Read_write } in
  let early_return = {name="early_return";args=[{name="p";value=writable_ptr}];
    body=[Const_f32(v,1.);Return None;Store_f32(writable_ptr,v)];threads_per_cta=None} in
  if not (contains (Ptx.emit early_return) "ret;\n  st.global.f32") then
    failwith "explicit return must remain before subsequent effects in PTX";
  let wrong_load_result=make_value 97 I32 and one=make_value 98 I32 and sum_i=make_value 99 I32 in
  let load_result_mismatch={name="load_result_mismatch";args=[{name="p";value=ro}];body=[
      Load_f32(wrong_load_result,ro);Const_i32(one,1);Add_i32(sum_i,wrong_load_result,one)];threads_per_cta=None} in
  expect_error "E_LOAD_TYPE" (Verifier.verify_kernel load_result_mismatch);
  let wrong_const=make_value 100 F32 in
  expect_error "E_CONST_TYPE" (Verifier.verify_kernel
    {name="wrong_const_result";args=[];body=[Const_i32(wrong_const,1)];threads_per_cta=None});
  let mask_index=make_value 52 I32 and wrong_index=make_value 53 I32
  and mask_bound=make_value 54 I32 in
  let masked_ptr=make_value ~permission:Read_only 55 (Ptr(F32,Gpu_type.Global))
  and masked_value=make_value 56 F32 in
  let bad_mask={name="bad_mask_index";args=[{name="p";value=ro}];body=[
      Const_i32(mask_index,0);Const_i32(wrong_index,1);Const_i32(mask_bound,2);
      Gep_f32(masked_ptr,ro,mask_index);
      Load_f32_masked(masked_value,masked_ptr,wrong_index,mask_bound)];threads_per_cta=None} in
  expect_error "E_MASK_INDEX" (Verifier.verify_kernel bad_mask);
  let good_mask={bad_mask with name="good_mask";body=[
      Const_i32(mask_index,0);Const_i32(mask_bound,2);
      Gep_f32(masked_ptr,ro,mask_index);
      Load_f32_masked(masked_value,masked_ptr,mask_index,mask_bound)]} in
  let masked_ptx=Ptx.emit good_mask in
  if not (contains masked_ptx "setp.ge.s32 %pred1, %r52, 0;" &&
          contains masked_ptx "and.pred %pred0, %pred0, %pred1;") then
    failwith "masked access PTX must reject negative indices";
  let forged = { ro with permission=Read_write } in
  let forged_store={readonly_store with body=[Const_f32(v,1.);Store_f32(forged,v)]} in
  expect_error "E_VALUE_IDENTITY" (Verifier.verify_kernel forged_store);
  let unique_root=make_value ~ownership:Unique ~provenance:(Some 70) 70 (Ptr(F32,Gpu_type.Global))
  and other_root=make_value ~provenance:(Some 71) 71 (Ptr(F32,Gpu_type.Global)) in
  let alias_kernel={name="derived_alias";args=[{name="u";value=unique_root};{name="a";value=other_root}];body=[];threads_per_cta=None} in
  let derived id root=make_value ~ownership:Aliased ~provenance:(Some root) id (Ptr(F32,Gpu_type.Global)) in
  let u0=derived 72 70 and u1=derived 73 70 and a0=derived 74 71 in
  if Verifier.may_alias alias_kernel u0 u1 |> not then failwith "same-root derived pointers must conservatively alias";
  if Verifier.may_alias alias_kernel u0 a0 then failwith "unique formal root should be noalias against distinct roots";
  let immutable=make_value ~permission:Immutable 80 (Ptr(F32,Gpu_type.Global)) in
  let immutable_load={name="immutable_load";args=[{name="p";value=immutable}];body=[Load_f32(make_value 81 F32,immutable)];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel immutable_load);
  let immutable_store={immutable_load with body=[Const_f32(v,1.);Store_f32(immutable,v)]} in
  expect_error "E_READONLY_STORE" (Verifier.verify_kernel immutable_store);
  let immutable_formal={name="immutable_formal";args=[{name="p";value=immutable}];body=[];threads_per_cta=None} in
  expect_error "E_ACCESS_REQUIRED" (Verifier.check_call immutable_formal [actual ~permission:Read_only 90]);
  expect_ok (Verifier.check_call {immutable_formal with args=[{name="p";value=ro}]} [actual ~permission:Immutable 90]);
  let reg_layout = Layout.Register {
    elements_per_lane=[1]; lanes_per_subgroup=[32]; subgroups_per_cta=[1]; order=[0] } in
  let blocked={Layout.elements_per_lane=[2;1];lanes_per_subgroup=[4;8];
    subgroups_per_cta=[2;1];order=[1;0]} in
  List.iter (fun coordinate ->
    let hardware=Layout.logical_to_hardware blocked coordinate in
    if Layout.hardware_to_logical blocked hardware <> coordinate then
      failwith "logical and hardware layout mappings must round-trip")
    [[0;0];[3;7];[15;7]];
  let interleaved={Layout.elements_per_lane=[2];lanes_per_subgroup=[32];
      subgroups_per_cta=[1];lane_mapping=Layout.Interleaved;
    lane_order=[0];register_order=[0];subgroup_order=[0]} in
  let map index=Layout.mapped_to_hardware interleaved [index] in
  if (map 0).lane_id <> 0 || (map 0).register_id <> 0 ||
     (map 1).lane_id <> 1 || (map 1).register_id <> 0 ||
     (map 32).lane_id <> 0 || (map 32).register_id <> 1 then
    failwith "interleaved layout must map each 32-element chunk across lanes";
  List.iter (fun index ->
    if Layout.mapped_hardware_to_logical interleaved (map index) <> [index] then
      failwith "mapped register coordinates must round-trip") [0;1;31;32;33;63];
  (match Layout.validate [Static 64] (Layout.Mapped_register interleaved) with
   | Ok () -> () | Error message -> failwith ("valid interleaved layout rejected: " ^ message));
  let permuted={Layout.elements_per_lane=[1;1];lanes_per_subgroup=[4;8];
      subgroups_per_cta=[1;1];lane_mapping=Layout.Blocked;
    lane_order=[1;0];register_order=[0;1];subgroup_order=[0;1]} in
  if (Layout.mapped_to_hardware permuted [1;2]).lane_id <> 10 then
    failwith "mapped lane dimension order must control hardware lane numbering";
  let smem_layout = Layout.Shared { vector_width=1; order=[0]; swizzle=Layout.No_swizzle } in
  let shape=[Static 32] in
  let input=make_value ~locality:Global ~permission:Read_only 60
      (MemRef(shape,Float32,Gpu_type.Global))
  and output=make_value ~ownership:Unique ~locality:Global ~permission:Read_write 61
      (MemRef(shape,Float32,Gpu_type.Global))
  and shared=make_value ~ownership:Unique ~locality:Local ~permission:Read_write
      ~layout:smem_layout 62 (MemRef(shape,Float32,Gpu_type.Shared))
  and from_global=make_value ~locality:Local ~layout:reg_layout 63
      (Tensor(shape,Float32))
  and from_shared=make_value ~locality:Local ~layout:reg_layout 64
      (Tensor(shape,Float32)) in
  let movement={name="shared_roundtrip";args=[{name="input";value=input};{name="output";value=output};
      ];body=[Shared_alloc shared;Load_tensor(from_global,input);Store_tensor(shared,from_global);
        Barrier Cta;Load_tensor(from_shared,shared);Store_tensor(output,from_shared)];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel movement);
  let scaled=make_value ~locality:Local ~layout:reg_layout 65 (Tensor(shape,Float32))
  and factor=make_value 66 F32 in
  let scale_kernel={movement with body=movement.body @ [Const_f32(factor,2.);
    Scale_tensor_f32(scaled,from_shared,factor)]} in
  expect_ok (Verifier.verify_kernel scale_kernel);
  let scale_ptx=Ptx.emit scale_kernel in
  if not (contains scale_ptx "mul.rn.f32 %f65, %f64, %f66;") then
    failwith "tile scaling PTX did not emit f32 multiplication";
  let int_shared=make_value ~permission:Read_write ~layout:smem_layout 101
      (MemRef(shape,Int32,Gpu_type.Shared))
  and int_tensor=make_value ~locality:Local ~layout:reg_layout 102
      (Tensor(shape,Int32))
  and scaled_int=make_value ~locality:Local ~layout:reg_layout 103
      (Tensor(shape,Int32))
  and int_factor=make_value 104 F32 in
  let bad_int_scale={name="bad_int_scale";args=[];body=[Shared_alloc int_shared;
      Load_tensor(int_tensor,int_shared);Const_f32(int_factor,2.);
      Scale_tensor_f32(scaled_int,int_tensor,int_factor)];threads_per_cta=None} in
  expect_error "E_TILE_ARITH_TYPE" (Verifier.verify_kernel bad_int_scale);
  expect_error "E_TILE_ARITH_TYPE" (Verifier.verify_kernel
    {bad_int_scale with body=[Shared_alloc int_shared;
      Load_tensor(int_tensor,int_shared);Mul_tensor_f32(scaled_int,int_tensor,int_tensor)]});
  let missing_barrier={movement with body=List.filter (function Barrier _ -> false | _ -> true) movement.body} in
  expect_error "E_SHARED_SYNC" (Verifier.verify_kernel missing_barrier);
  let bad_layout=Layout.Register {
    elements_per_lane=[1]; lanes_per_subgroup=[32]; subgroups_per_cta=[1]; order=[1] } in
  let malformed={movement with body=[Load_tensor({from_global with layout=Some bad_layout},input)]} in
  expect_error "E_LAYOUT_INVALID" (Verifier.verify_kernel malformed);
  let global_with_register_layout={movement with
    args=[{name="input";value={input with layout=Some reg_layout}};{name="output";value=output}];body=[]} in
  expect_error "E_MEMREF_LAYOUT" (Verifier.verify_kernel global_with_register_layout);
  let movement_ptx=Ptx.emit movement in
  let movement_target=Ptx_lowering.lower movement in
  if movement_target.Ptx_ir.launch.threads_per_cta <> Some (32,1,1) ||
     movement_target.Ptx_ir.launch.shared_bytes <> 128 then
    failwith "PTX target lowering must derive exact CTA and shared-memory requirements";
  let movement_physical=Physical_ir.physicalize movement_target in
  let lane17=Physical_ir.coordinate movement_physical from_global [17] in
  if lane17.Layout.lane_id <> 17 || lane17.register_id <> 0 then
    failwith "physical tensor values must retain their logical-to-hardware mapping";
  let small_shared=make_value ~permission:Read_write
      ~layout:(Layout.Shared {vector_width=1;order=[0];swizzle=Layout.No_swizzle}) 120
      (MemRef([Static 16],Float32,Gpu_type.Shared))
  and large_shared=make_value ~permission:Read_write
      ~layout:(Layout.Shared {vector_width=1;order=[0];swizzle=Layout.No_swizzle}) 121
      (MemRef([Static 64],Int32,Gpu_type.Shared)) in
  let differently_sized_shared={name="different_shared_sizes";args=[];
    body=[Shared_alloc small_shared;Shared_alloc large_shared];threads_per_cta=None} in
  let sized_target=Ptx_lowering.lower differently_sized_shared in
  (* One window: 64 bytes padded to the 128-byte window alignment, then 256. *)
  if sized_target.Ptx_ir.launch.shared_bytes <> 384 then
    failwith "shared launch size must be the whole window, including alignment padding";
  if (Ptx_ir.shared_plan "different_shared_sizes" sized_target.Ptx_ir.body).Ptx_ir.offsets
     <> [(120, 0); (121, 128)] then
    failwith "shared allocations must be laid out at distinct aligned window offsets";
  let sized_ptx=Ptx.emit_target sized_target in
  if not (contains sized_ptx ".shared .align 128 .b8 __oxgpu_smem[384];") then
    failwith "a within-cap shared window must stay statically declared";
  if contains sized_ptx ".extern .shared" then
    failwith "a within-cap shared window must not use the dynamic window";
  (* Over the 48 KiB static cap the window has to become the dynamic one, and
     the emitted PTX has to say how many bytes a launch must request. *)
  let big_shared=make_value ~permission:Read_write
      ~layout:(Layout.Shared {vector_width=1;order=[0];swizzle=Layout.No_swizzle}) 130
      (MemRef([Static 20000],Float32,Gpu_type.Shared)) in
  let big_kernel={name="big_shared";args=[];
    body=[Shared_alloc big_shared];threads_per_cta=None} in
  let big_target=Ptx_lowering.lower big_kernel in
  if big_target.Ptx_ir.launch.shared_bytes <> 80000 then
    failwith "an over-cap shared window must still be sized exactly";
  let big_ptx=Ptx.emit_target big_target in
  if not (contains big_ptx ".extern .shared .align 128 .b8 __oxgpu_smem[];"
          && contains big_ptx "// oxgpu.shared.dynamic 80000") then
    failwith "an over-cap shared window must be dynamic and record its launch size";
  (match Ptx_ir.shared_plan "over_budget"
           [Shared_alloc (make_value ~permission:Read_write
             ~layout:(Layout.Shared {vector_width=1;order=[0];swizzle=Layout.No_swizzle}) 131
             (MemRef([Static 70000],Float32,Gpu_type.Shared)))] with
   | _ -> failwith "shared allocations beyond an SM's capacity must be rejected"
   | exception Invalid_argument _ -> ());
  let padded_shared=make_value ~permission:Read_write
      ~layout:(Layout.Padded_shared {
        base={vector_width=1;order=[1;0];swizzle=Layout.No_swizzle};row_padding_bytes=16 }) 122
      (MemRef([Static 4;Static 8],Float32,Gpu_type.Shared)) in
  let padded_kernel={name="padded_shared";args=[];body=[Shared_alloc padded_shared];threads_per_cta=None} in
  let padded_target=Ptx_lowering.lower padded_kernel in
  if padded_target.Ptx_ir.launch.shared_bytes <> 192 ||
     not (contains (Ptx.emit_target padded_target) "__oxgpu_smem[192]") then
    failwith "shared row padding must affect the allocation and launch contract";
  List.iter (fun part -> if not (contains movement_ptx part) then
    failwith ("shared movement PTX missing " ^ part))
    [".shared .align 128 .b8 __oxgpu_smem[128];";
     ".reqntid 32, 1, 1";"ld.global.f32";"st.shared.f32";"bar.sync 0;";"ld.shared.f32";"st.global.f32"];
  let x=make_value ~locality:Global ~permission:Read_only 90 (MemRef(shape,Float32,Gpu_type.Global))
  and y=make_value ~locality:Global ~permission:Read_only 91 (MemRef(shape,Float32,Gpu_type.Global))
  and result=make_value ~ownership:Unique ~locality:Global ~permission:Write_only 92
      (MemRef([Static 1],Float32,Gpu_type.Global)) in
  let tx=make_value ~locality:Local ~layout:reg_layout 93 (Tensor(shape,Float32))
  and ty=make_value ~locality:Local ~layout:reg_layout 94 (Tensor(shape,Float32))
  and products=make_value ~locality:Local ~layout:reg_layout 95 (Tensor(shape,Float32))
  and sum=make_value 96 F32 in
  let high_level_dot={name="dot_ir";args=[{name="x";value=x};{name="y";value=y};{name="result";value=result}];body=[
      Load_tensor(tx,x);Load_tensor(ty,y);Mul_tensor_f32(products,tx,ty);
      Reduce_sum_f32(sum,products);Store_f32_grid_leader(result,sum)];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel high_level_dot);
  let dot_ptx=Ptx.emit high_level_dot in
  let dot_target=Ptx_lowering.lower high_level_dot in
  if not (List.exists (function Ptx_ir.Tensor_mul_f32 _ -> true | _ -> false) dot_target.body) then
    failwith "tensor multiply must retain its semantic operation in target IR";
  (try
     ignore (Ptx.emit_target {dot_target with body=[Ptx_ir.Mul_f32(products,tx,ty)]});
     failwith "PTX scalar multiply accepted tensor values"
   with Invalid_argument message when contains message "scalar f32 operands" -> ());
  List.iter (fun part -> if not (contains dot_ptx part) then
    failwith ("lowered dot PTX missing " ^ part))
    [".reqntid 32, 1, 1";"ld.global.f32";"mul.rn.f32";"shfl.sync.bfly.b32";"st.global.f32"];
  let unrelated_global_index=make_value 130 I32 in
  let independent_index_and_reduction={high_level_dot with name="independent_index_reduction";
    body=Global_idx_x unrelated_global_index :: high_level_dot.body} in
  ignore (Ptx_lowering.lower independent_index_and_reduction);
  let one_warp_mapped={Layout.elements_per_lane=[1];lanes_per_subgroup=[32];
    subgroups_per_cta=[1];lane_mapping=Layout.Interleaved;lane_order=[0];
    register_order=[0];subgroup_order=[0]} in
  let mapped_tensor id=make_value ~locality:Local
      ~layout:(Layout.Mapped_register one_warp_mapped) id (Tensor(shape,Float32)) in
  let mapped_x=mapped_tensor 140 and mapped_y=mapped_tensor 141
  and mapped_products=mapped_tensor 142 and mapped_sum=make_value 143 F32 in
  let mapped_dot={high_level_dot with name="mapped_dot";body=[
      Load_tensor(mapped_x,x);Load_tensor(mapped_y,y);Mul_tensor_f32(mapped_products,mapped_x,mapped_y);
      Reduce_sum_f32(mapped_sum,mapped_products);Store_f32_grid_leader(result,mapped_sum)]} in
  expect_ok (Verifier.verify_kernel mapped_dot);
  ignore (Compiler.compile_ptx mapped_dot);
  let wide_shape=[Static 64] in
  let wide_mem=make_value ~locality:Global ~permission:Read_only 131
      (MemRef(wide_shape,Float32,Gpu_type.Global))
  and wide_tensor=make_value ~locality:Local ~layout:(Layout.Mapped_register interleaved) 132
      (Tensor(wide_shape,Float32)) in
  let wide_tile={name="wide_tile";args=[{name="src";value=wide_mem}];
    body=[Load_tensor(wide_tensor,wide_mem)];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel wide_tile);
  let wide_rhs=make_value ~locality:Global ~permission:Read_only 136
      (MemRef(wide_shape,Float32,Gpu_type.Global))
  and wide_out=make_value ~ownership:Unique ~locality:Global ~permission:Read_write 133
      (MemRef(wide_shape,Float32,Gpu_type.Global))
  and wide_scaled=make_value ~locality:Local ~layout:(Layout.Mapped_register interleaved) 134
      (Tensor(wide_shape,Float32))
  and wide_rhs_tensor=make_value ~locality:Local ~layout:(Layout.Mapped_register interleaved) 137
      (Tensor(wide_shape,Float32))
  and wide_product=make_value ~locality:Local ~layout:(Layout.Mapped_register interleaved) 138
      (Tensor(wide_shape,Float32))
  and wide_factor=make_value 135 F32 in
  let wide_kernel={name="wide_interleaved";
    args=[{name="src";value=wide_mem};{name="rhs";value=wide_rhs};{name="dst";value=wide_out}];
    body=[Load_tensor(wide_tensor,wide_mem);Load_tensor(wide_rhs_tensor,wide_rhs);Const_f32(wide_factor,2.);
      Scale_tensor_f32(wide_scaled,wide_tensor,wide_factor);
      Mul_tensor_f32(wide_product,wide_scaled,wide_rhs_tensor);Store_tensor(wide_out,wide_product)];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel wide_kernel);
  let wide_ptx=Compiler.compile_ptx wide_kernel in
  if count wide_ptx "ld.global.f32" <> 4 || count wide_ptx "st.global.f32" <> 2 ||
     count wide_ptx "mul.rn.f32" <> 4 || not (contains wide_ptx ", 128;") then
    failwith "interleaved two-register tiles must load, scale, multiply, and store each lane value";
  let blocked_layout=Layout.Mapped_register {interleaved with lane_mapping=Layout.Blocked} in
  let blocked_value value={value with layout=Some blocked_layout} in
  let blocked_tensor=blocked_value wide_tensor and blocked_rhs=blocked_value wide_rhs_tensor
  and blocked_scaled=blocked_value wide_scaled and blocked_product=blocked_value wide_product in
  let blocked_kernel={wide_kernel with name="wide_blocked";body=[
      Load_tensor(blocked_tensor,wide_mem);Load_tensor(blocked_rhs,wide_rhs);Const_f32(wide_factor,2.);
      Scale_tensor_f32(blocked_scaled,blocked_tensor,wide_factor);
      Mul_tensor_f32(blocked_product,blocked_scaled,blocked_rhs);Store_tensor(wide_out,blocked_product)]} in
  expect_ok (Verifier.verify_kernel blocked_kernel);
  let blocked_ptx=Compiler.compile_ptx blocked_kernel in
  if count blocked_ptx "ld.global.f32" <> 4 || count blocked_ptx "st.global.f32" <> 2 ||
     not (contains blocked_ptx ", 8;") || not (contains blocked_ptx ", 4;") then
    failwith "blocked two-register mapping must use adjacent elements per lane";
  let wide_sum=make_value 139 F32 in
  let wide_reduction={name="wide_reduction";
    args=[{name="src";value=wide_mem}];body=[Load_tensor(wide_tensor,wide_mem);
      Reduce_sum_f32(wide_sum,wide_tensor)];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel wide_reduction);
  let reduction_ptx=Compiler.compile_ptx wide_reduction in
  if not (contains reduction_ptx "add.rn.f32" && contains reduction_ptx "shfl.sync.bfly.b32") then
    failwith "wide reduction must sum local register slots before the warp reduction";
  List.iter (fun part -> if not (contains dot_ptx part) then
    failwith ("grid-leader predicate missing " ^ part)) ["%tid.y";"%tid.z"];
  let alias_sensitive_kernel y_ownership =
    let source=make_value ~provenance:(Some 110) ~permission:Read_only 110
        (Ptr(F32,Gpu_type.Global))
    and destination=make_value ~ownership:y_ownership ~provenance:(Some 111)
        ~permission:Read_write 111 (Ptr(F32,Gpu_type.Global))
    and loaded=make_value 112 F32 and replacement=make_value 113 F32
    and reloaded=make_value 114 F32 in
    {name="alias_sensitive_load";
     args=[{name="x";value=source};{name="y";value=destination}];
     body=[Load_f32(loaded,source);Const_f32(replacement,0.);
       Store_f32(destination,replacement);Load_f32(reloaded,source);
       Store_f32(destination,reloaded)];threads_per_cta=None} in
  let aliased_ptx=Compiler.compile_ptx (alias_sensitive_kernel Aliased)
  and unique_ptx=Compiler.compile_ptx (alias_sensitive_kernel Unique) in
  if count aliased_ptx "ld.global.f32" <> 2 then
    failwith "aliased store incorrectly preserved a cached global load";
  if count unique_ptx "ld.global.f32" <> 1 then
    failwith "unique noalias fact did not eliminate the redundant global load";
  if meet_locality Global Local <> Local || meet_locality Global Global <> Global then
    failwith "locality meet must keep the weaker lifetime";
  if meet_permission Immutable Read_only <> Read_only ||
     meet_permission Read_write Read_write <> Read_write then
    failwith "permission meet must not invent stronger access";
  if meet_gpu_boundary Boundary_portable Boundary_local <> Boundary_unspecified ||
     meet_gpu_boundary Boundary_portable Boundary_portable <> Boundary_portable then
    failwith "gpu-boundary meet must agree or fall back to unspecified";
  let buffer_slot =
    { Kernel_ast.loc=Source_span.synthetic; ty=MemRef([Dynamic],Float32,Gpu_type.Global);
      ownership=Unique; locality=Global; domain_portability=Domain_nonportable;
      gpu_boundary=Boundary_unspecified; permission=Read_write } in
  let scalar_slot =
    { Kernel_ast.loc=Source_span.synthetic; ty=F32; ownership=Aliased; locality=Global;
      domain_portability=Domain_nonportable; gpu_boundary=Boundary_unspecified;
      permission=Read_only } in
  let stamped=Kernel_frontend.lower {
    name="boundary_stamp";
    args=[{index=0;slot=buffer_slot};{index=1;slot=scalar_slot}];
    result={buffer_slot with ty=Unit; ownership=Aliased; permission=Read_only};
    body=[];
    threads=None} in
  let buf=(List.nth stamped.args 0).value and scalar=(List.nth stamped.args 1).value in
  if buf.gpu_boundary <> Boundary_portable then
    failwith "buffer formals must derive Boundary_portable at kernel entry";
  if scalar.gpu_boundary <> Boundary_unspecified then
    failwith "scalar formals must keep Boundary_unspecified";
  expect_ok (Launch_contract.check stamped [
    Launch_contract.buffer ~ownership:Unique ~permission:Read_write 0;
    Launch_contract.buffer ~permission:Read_only ~gpu_boundary:Boundary_unspecified 1]);
  expect_error "E_GPU_BOUNDARY_REQUIRED"
    (Launch_contract.check stamped [
      Launch_contract.buffer ~ownership:Unique ~permission:Read_write
        ~gpu_boundary:Boundary_unspecified 0;
      Launch_contract.buffer ~permission:Read_only ~gpu_boundary:Boundary_unspecified 1]);
  expect_error "E_UNIQUE_REQUIRED"
    (Launch_contract.check stamped [
      Launch_contract.buffer ~permission:Read_write 0;
      Launch_contract.buffer ~permission:Read_only ~gpu_boundary:Boundary_unspecified 1]);
  let join_kernel_ast : Kernel_ast.t = {
    name="join_meet";
    args=[];
    threads=None;
    result={loc=Source_span.synthetic;ty=Unit;ownership=Aliased;locality=Global;
      domain_portability=Domain_nonportable;gpu_boundary=Boundary_unspecified;
      permission=Read_only};
    body=[
      Kernel_ast.instruction (Const_bool(0,true));
      Kernel_ast.instruction (If(Some(3,F32),Value 0,
        {body=[Kernel_ast.instruction (Const_f32(1,1.))]; yield=Some(Value 1)},
        {body=[Kernel_ast.instruction (Const_f32(2,2.))]; yield=Some(Value 2)}))]} in
  let joined=Kernel_frontend.lower join_kernel_ast in
  (match List.find_map (function If(Some dst,_,_,_) -> Some dst | _ -> None) joined.body with
   | Some dst when dst.ownership=Aliased && dst.locality=Local &&
       dst.permission=Read_only && dst.provenance=None -> ()
   | _ -> failwith "branch join must meet to aliased local readable scalar facts");
  let shape2=[Static 4; Static 8] in
  let layout2=Layout.Mapped_register {
    elements_per_lane=[1;1]; lanes_per_subgroup=[4;8]; subgroups_per_cta=[1;1];
    lane_mapping=Layout.Blocked; lane_order=[1;0]; register_order=[1;0]; subgroup_order=[1;0]} in
  let mem2=make_value ~locality:Global ~permission:Read_only 150
      (MemRef(shape2,Float32,Gpu_type.Global))
  and out2=make_value ~ownership:Unique ~locality:Global ~permission:Write_only 151
      ~layout:(Layout.Global {strides=[16;1]}) (MemRef(shape2,Float32,Gpu_type.Global))
  and tile2=make_value ~locality:Local ~layout:layout2 152
      (Tensor(shape2,Float32))
  and rows=make_value 153 I32 and cols=make_value 154 I32 in
  let tile2d={name="tile2d_strided";
    args=[{name="src";value=mem2};{name="dst";value=out2}];
    body=[Load_tensor(tile2,mem2);Store_tensor(out2,tile2)];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel tile2d);
  let tile2d_ptx=Compiler.compile_ptx tile2d in
  if not (contains tile2d_ptx "ld.global.f32" && contains tile2d_ptx "st.global.f32" &&
          contains tile2d_ptx ".reqntid 32, 1, 1") then
    failwith "2D strided tile movement must lower to global loads/stores under one warp";
  let masked2d={name="tile2d_masked";
    args=[{name="src";value=mem2};{name="dst";value=out2}];
    body=[Const_i32(rows,3);Const_i32(cols,5);Load_tensor_masked(tile2,mem2,rows,cols);
      Store_tensor_masked(out2,tile2,rows,cols)];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel masked2d);
  let masked_ptx=Compiler.compile_ptx masked2d in
  if not (contains masked_ptx "setp.lt.s32" && contains masked_ptx "@%pred0 ld.global.f32" &&
          contains masked_ptx "@%pred0 st.global.f32") then
    failwith "masked 2D tile movement must predicate row/col bounds";
  let remap_src_layout=Layout.Mapped_register {
    elements_per_lane=[2]; lanes_per_subgroup=[32]; subgroups_per_cta=[1];
    lane_mapping=Layout.Blocked; lane_order=[0]; register_order=[0]; subgroup_order=[0]} in
  let remap_dst_layout=Layout.Mapped_register {
    elements_per_lane=[2]; lanes_per_subgroup=[32]; subgroups_per_cta=[1];
    lane_mapping=Layout.Interleaved; lane_order=[0]; register_order=[0]; subgroup_order=[0]} in
  let remap_mem=make_value ~locality:Global ~permission:Read_write 160
      (MemRef([Static 64], Float32, Gpu_type.Global))
  and remap_src=make_value ~locality:Local ~layout:remap_src_layout 161
      (Tensor([Static 64], Float32))
  and remap_dst=make_value ~locality:Local ~layout:remap_dst_layout 162
      (Tensor([Static 64], Float32)) in
  let remap_kernel={name="remap_tile";args=[{name="buf";value=remap_mem}];
    body=[Load_tensor(remap_src,remap_mem);Remap_tensor(remap_dst,remap_src);
      Store_tensor(remap_mem,remap_dst)];threads_per_cta=None} in
  expect_ok (Verifier.verify_kernel remap_kernel);
  let remap_ptx=Compiler.compile_ptx remap_kernel in
  if not (contains remap_ptx "st.shared.f32" && contains remap_ptx "bar.sync 0;" &&
          contains remap_ptx "ld.shared.f32") then
    failwith "tensor remap must stage through shared memory with a CTA barrier";
  (* Explicit shared staging + MAD — no Matmul_f32 compiler expansion. *)
  let sm_layout =
    Layout.Shared { vector_width = 1; order = [0]; swizzle = Layout.No_swizzle }
  in
  let a_sm =
    make_value ~permission:Read_write ~layout:sm_layout 170
      (MemRef ([Static 32], Float32, Shared))
  and acc = make_value ~locality:Local ~permission:Read_write 171 F32
  and aval = make_value ~locality:Local 172 F32
  and bval = make_value ~locality:Local 173 F32
  and zero = make_value ~locality:Local 174 I32
  and out = make_value ~locality:Local 175 F32 in
  let explicit_gemm =
    { name = "explicit_shared_mad"
    ; args = []
    ; body =
        [ Shared_alloc a_sm
        ; Const_i32 (zero, 0)
        ; Const_f32 (aval, 1.0)
        ; Shared_store_f32 (a_sm, aval, zero)
        ; Barrier Execution.Cta
        ; Shared_load_f32 (bval, a_sm, zero)
        ; Const_f32 (acc, 0.)
        ; Mad_f32 (out, bval, aval, acc)
        ]
    ; threads_per_cta = Some 32
    }
  in
  expect_ok (Verifier.verify_kernel explicit_gemm);
  let gemm_ptx = Compiler.compile_ptx explicit_gemm in
  if not (contains gemm_ptx "ld.shared.f32" && contains gemm_ptx "fma.rn.f32"
          && contains gemm_ptx "bar.sync 0;") then
    failwith "explicit shared+MAD IR must lower without a matmul op";
  (* --- automatic store vectorization ---
     Consecutive scalar stores fuse only when the indices are provably
     consecutive and the first is provably a multiple of the width. Each
     negative case below must stay scalar. *)
  let store_kernel name prelude indices =
    let buffer=make_value ~locality:Global ~permission:Read_write 300
        (MemRef([Static 64],Float32,Gpu_type.Global)) in
    let one=make_value 301 F32 in
    let stores=List.concat (List.mapi (fun i index ->
      let address=make_value ~locality:Global ~permission:Read_write (400+i)
          (Ptr (F32, Gpu_type.Global)) in
      [Gep_f32 (address, buffer, index); Store_f32 (address, one)]) indices) in
    { name; args=[{name="y";value=buffer}];
      body=prelude @ [Const_f32 (one, 1.)] @ stores; threads_per_cta=Some 32 }
  in
  let integer id = make_value id I32 in
  let constants values =
    let instructions=List.mapi (fun i n -> Const_i32 (integer (310+i), n)) values in
    (instructions, List.mapi (fun i _ -> integer (310+i)) values)
  in
  let count needle text =
    let n=String.length needle in
    let rec loop i total =
      if i + n > String.length text then total
      else loop (i+1) (if String.sub text i n = needle then total+1 else total) in
    loop 0 0
  in
  let expect_stores name kernel ~v4 ~v2 ~scalar =
    let ptx=Compiler.compile_ptx kernel in
    if count "st.global.v4.f32" ptx <> v4 || count "st.global.v2.f32" ptx <> v2
       || count "st.global.f32" ptx <> scalar then
      failwith (Printf.sprintf
        "%s: expected %d v4 / %d v2 / %d scalar stores, got %d / %d / %d" name
        v4 v2 scalar (count "st.global.v4.f32" ptx) (count "st.global.v2.f32" ptx)
        (count "st.global.f32" ptx))
  in
  (* Four consecutive constant indices from zero: one 16-byte store. *)
  let prelude, indices = constants [0;1;2;3] in
  expect_stores "aligned quad" (store_kernel "aligned_quad" prelude indices)
    ~v4:1 ~v2:0 ~scalar:0;
  (* Consecutive, but the run starts at 1, so no width divides it. *)
  let prelude, indices = constants [1;2] in
  expect_stores "misaligned pair" (store_kernel "misaligned_pair" prelude indices)
    ~v4:0 ~v2:0 ~scalar:2;
  (* Aligned, but the indices skip one. *)
  let prelude, indices = constants [0;2] in
  expect_stores "strided pair" (store_kernel "strided_pair" prelude indices)
    ~v4:0 ~v2:0 ~scalar:2;
  (* A dynamic base whose parity is unknown cannot be proved aligned. *)
  let tid=integer 320 and one_i=integer 321 and tid1=integer 322 in
  expect_stores "unknown parity"
    (store_kernel "unknown_parity"
       [Thread_idx_x tid; Const_i32 (one_i, 1); Add_i32 (tid1, tid, one_i)]
       [tid; tid1])
    ~v4:0 ~v2:0 ~scalar:2;
  (* The same base rounded down to a multiple of two is provably even, so the
     pair fuses even though its value is only known at run time. This is the
     matmul epilogue's shape: an unknown row times a stride known to be a
     multiple of the tile width. *)
  let tid=integer 330 and two=integer 331 and half=integer 332
  and even=integer 333 and one_i=integer 334 and even1=integer 335 in
  expect_stores "proven even base"
    (store_kernel "proven_even_base"
       [Thread_idx_x tid; Const_i32 (two, 2); Div_i32 (half, tid, two);
        Mul_i32 (even, half, two); Const_i32 (one_i, 1);
        Add_i32 (even1, even, one_i)]
       [even; even1])
    ~v4:0 ~v2:1 ~scalar:0;
  (* Fusing moves the earlier store later, so anything between the two that
     touches memory blocks the run. A load is the smallest such case. (A
     foreign *store* in between is already blocked earlier: runs are built
     from candidates adjacent in program order, so that store would break the
     chain by base or index before this check is reached.) *)
  let interposed =
    let buffer=make_value ~locality:Global ~permission:Read_write 300
        (MemRef([Static 64],Float32,Gpu_type.Global)) in
    let one=make_value 341 F32 in
    let address id = make_value ~locality:Global ~permission:Read_write id
        (Ptr (F32, Gpu_type.Global)) in
    let zero=integer 342 and one_i=integer 343 and three=integer 344 in
    let p0=address 350 and p1=address 351 and p2=address 352 in
    let loaded=make_value 353 F32 in
    { name="interposed_load";
      args=[{name="y";value=buffer}];
      body=[Const_i32 (zero, 0); Const_i32 (one_i, 1); Const_i32 (three, 3);
            Const_f32 (one, 1.);
            Gep_f32 (p0, buffer, zero); Store_f32 (p0, one);
            Gep_f32 (p2, buffer, three); Load_f32 (loaded, p2);
            Gep_f32 (p1, buffer, one_i); Store_f32 (p1, loaded)];
      threads_per_cta=Some 32 }
  in
  expect_stores "interposed load" interposed ~v4:0 ~v2:0 ~scalar:2;
  print_endline "verifier semantic tests passed"
