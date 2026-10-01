open Gpu_type
open Execution
open Mode
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
  let kernel={name="alias_check";args=[{name="x";value=x};{name="y";value=y}];body=[]} in
  expect_ok (Verifier.verify_kernel kernel);
  expect_ok (Verifier.check_call kernel [actual 10;actual ~ownership:Unique ~permission:Read_write 11]);
  expect_error "E_UNIQUE_ALIAS" (Verifier.check_call kernel [actual 10;actual ~ownership:Unique ~permission:Read_write 10]);
  expect_error "E_UNIQUE_REQUIRED" (Verifier.check_call kernel [actual 10;actual ~permission:Read_write 11]);
  let modal_arg=make_value ~locality:Global ~domain_portability:Domain_portable
      ~gpu_boundary:Boundary_portable 105 (Ptr(F32,Gpu_type.Global)) in
  let modal_kernel={name="modal_call";args=[{name="p";value=modal_arg}];body=[]} in
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
  let nonunit_signature : Oxcaml_frontend.signature = {
    name="nonunit"; args=[];
    result={ty=F32; ownership=Aliased; locality=Global;
      domain_portability=Domain_nonportable; gpu_boundary=Boundary_unspecified;
      permission=Read_only} } in
  (try
     ignore (Kernel_frontend.lower {Kernel_ast.name="nonunit";args=[];body=[]} nonunit_signature);
     failwith "non-unit GPU kernel result unexpectedly lowered"
   with Invalid_argument message when message = "GPU kernel entry points must return unit" -> ());
  let local = make_value ~locality:Local 40 (Ptr(F32,Gpu_type.Global)) in
  let returning = { kernel with name="bad_escape"; body=[Return(Some local)] } in
  expect_error "E_KERNEL_RETURN_TYPE" (Verifier.verify_kernel returning);
  let ro=make_value ~permission:Read_only 50 (Ptr(F32,Gpu_type.Global)) and v=make_value 51 F32 in
  let readonly_store={name="bad_store";args=[{name="p";value=ro}];body=[Const_f32(v,1.);Store_f32(ro,v)]} in
  expect_error "E_READONLY_STORE" (Verifier.verify_kernel readonly_store);
  let writable_ptr = { ro with permission=Read_write } in
  let early_return = {name="early_return";args=[{name="p";value=writable_ptr}];
    body=[Const_f32(v,1.);Return None;Store_f32(writable_ptr,v)]} in
  if not (contains (Ptx.emit early_return) "ret;\n  st.global.f32") then
    failwith "explicit return must remain before subsequent effects in PTX";
  let wrong_load_result=make_value 97 I32 and one=make_value 98 I32 and sum_i=make_value 99 I32 in
  let load_result_mismatch={name="load_result_mismatch";args=[{name="p";value=ro}];body=[
      Load_f32(wrong_load_result,ro);Const_i32(one,1);Add_i32(sum_i,wrong_load_result,one)]} in
  expect_error "E_LOAD_TYPE" (Verifier.verify_kernel load_result_mismatch);
  let wrong_const=make_value 100 F32 in
  expect_error "E_CONST_TYPE" (Verifier.verify_kernel
    {name="wrong_const_result";args=[];body=[Const_i32(wrong_const,1)]});
  let mask_index=make_value 52 I32 and wrong_index=make_value 53 I32
  and mask_bound=make_value 54 I32 in
  let masked_ptr=make_value ~permission:Read_only 55 (Ptr(F32,Gpu_type.Global))
  and masked_value=make_value 56 F32 in
  let bad_mask={name="bad_mask_index";args=[{name="p";value=ro}];body=[
      Const_i32(mask_index,0);Const_i32(wrong_index,1);Const_i32(mask_bound,2);
      Gep_f32(masked_ptr,ro,mask_index);
      Load_f32_masked(masked_value,masked_ptr,wrong_index,mask_bound)]} in
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
  let alias_kernel={name="derived_alias";args=[{name="u";value=unique_root};{name="a";value=other_root}];body=[]} in
  let derived id root=make_value ~ownership:Aliased ~provenance:(Some root) id (Ptr(F32,Gpu_type.Global)) in
  let u0=derived 72 70 and u1=derived 73 70 and a0=derived 74 71 in
  if Verifier.may_alias alias_kernel u0 u1 |> not then failwith "same-root derived pointers must conservatively alias";
  if Verifier.may_alias alias_kernel u0 a0 then failwith "unique formal root should be noalias against distinct roots";
  let immutable=make_value ~permission:Immutable 80 (Ptr(F32,Gpu_type.Global)) in
  let immutable_load={name="immutable_load";args=[{name="p";value=immutable}];body=[Load_f32(make_value 81 F32,immutable)]} in
  expect_ok (Verifier.verify_kernel immutable_load);
  let immutable_store={immutable_load with body=[Const_f32(v,1.);Store_f32(immutable,v)]} in
  expect_error "E_READONLY_STORE" (Verifier.verify_kernel immutable_store);
  let immutable_formal={name="immutable_formal";args=[{name="p";value=immutable}];body=[]} in
  expect_error "E_ACCESS_REQUIRED" (Verifier.check_call immutable_formal [actual ~permission:Read_only 90]);
  expect_ok (Verifier.check_call {immutable_formal with args=[{name="p";value=ro}]} [actual ~permission:Immutable 90]);
  let reg_layout = Layout.Register {
    elements_per_lane=[1]; lanes_per_subgroup=[32]; subgroups_per_cta=[1]; order=[0] } in
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
        Barrier Cta;Load_tensor(from_shared,shared);Store_tensor(output,from_shared)]} in
  expect_ok (Verifier.verify_kernel movement);
  let scaled=make_value ~locality:Local ~layout:reg_layout 65 (Tensor(shape,Float32))
  and factor=make_value 66 F32 in
  let scale_kernel={movement with body=movement.body @ [Const_f32(factor,2.);
    Scale_tensor_f32(scaled,from_shared,factor)]} in
  expect_ok (Verifier.verify_kernel scale_kernel);
  let scale_ptx=Ptx.emit scale_kernel in
  if not (contains scale_ptx "mul.f32 %f65, %f64, %f66;") then
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
      Scale_tensor_f32(scaled_int,int_tensor,int_factor)]} in
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
  let movement_target=Ptx.lower_to_ptx_ir movement in
  if movement_target.Ptx_ir.launch.threads_per_cta <> Some (32,1,1) ||
     movement_target.Ptx_ir.launch.static_shared_bytes <> 128 then
    failwith "PTX target lowering must derive exact CTA and static shared-memory requirements";
  List.iter (fun part -> if not (contains movement_ptx part) then
    failwith ("shared movement PTX missing " ^ part))
    [".shared .align 16 .b8 __shared_62[128];";
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
      Reduce_sum_f32(sum,products);Store_f32_grid_leader(result,sum)]} in
  expect_ok (Verifier.verify_kernel high_level_dot);
  let dot_ptx=Ptx.emit high_level_dot in
  List.iter (fun part -> if not (contains dot_ptx part) then
    failwith ("lowered dot PTX missing " ^ part))
    [".reqntid 32, 1, 1";"ld.global.f32";"mul.f32";"shfl.sync.bfly.b32";"st.global.f32"];
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
       Store_f32(destination,reloaded)]} in
  let aliased_ptx=Compiler.compile_ptx (alias_sensitive_kernel Aliased)
  and unique_ptx=Compiler.compile_ptx (alias_sensitive_kernel Unique) in
  if count aliased_ptx "ld.global.f32" <> 2 then
    failwith "aliased store incorrectly preserved a cached global load";
  if count unique_ptx "ld.global.f32" <> 1 then
    failwith "unique noalias fact did not eliminate the redundant global load";
  print_endline "verifier semantic tests passed"
