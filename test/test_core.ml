open Gpu_type
open Mode
open Ir

let expect_ok = function Ok () -> () | Error es -> failwith (String.concat "; " (List.map (fun e -> e.Verifier.code ^ ": " ^ e.message) es))
let expect_error code = function
  | Error es when List.exists (fun e -> e.Verifier.code = code) es -> ()
  | Error es -> failwith ("expected " ^ code ^ ", got " ^ String.concat "," (List.map (fun e -> e.Verifier.code) es))
  | Ok () -> failwith ("expected error " ^ code)
let actual ?(ownership=Aliased) ?(permission=Read_only) buffer_id =
  { buffer_id; actual_ownership=ownership; actual_permission=permission }
let contains text needle =
  let n=String.length needle in
  let rec loop i = i + n <= String.length text &&
    (String.sub text i n = needle || loop (i+1)) in loop 0

let () =
  let x=make_value ~locality:Global ~permission:Read_only 0 (Ptr(F32,Gpu_type.Global))
  and y=make_value ~ownership:Unique ~locality:Global ~permission:Read_write 1 (Ptr(F32,Gpu_type.Global)) in
  let kernel={name="alias_check";args=[{name="x";value=x};{name="y";value=y}];body=[]} in
  expect_ok (Verifier.verify_kernel kernel);
  expect_ok (Verifier.check_call kernel [actual 10;actual ~ownership:Unique ~permission:Read_write 11]);
  expect_error "E_UNIQUE_ALIAS" (Verifier.check_call kernel [actual 10;actual ~ownership:Unique ~permission:Read_write 10]);
  expect_error "E_UNIQUE_REQUIRED" (Verifier.check_call kernel [actual 10;actual ~permission:Read_write 11]);
  let local = make_value ~locality:Local 40 (Ptr(F32,Gpu_type.Global)) in
  let returning = { kernel with name="bad_escape"; body=[Return(Some local)] } in
  expect_error "E_LOCAL_ESCAPE" (Verifier.verify_kernel returning);
  let ro=make_value ~permission:Read_only 50 (Ptr(F32,Gpu_type.Global)) and v=make_value 51 F32 in
  let readonly_store={name="bad_store";args=[{name="p";value=ro}];body=[Const_f32(v,1.);Store_f32(ro,v)]} in
  expect_error "E_READONLY_STORE" (Verifier.verify_kernel readonly_store);
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
    size_per_thread=[1]; threads_per_warp=[32]; warps_per_cta=[1]; order=[0] } in
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
  let missing_barrier={movement with body=List.filter (function Barrier _ -> false | _ -> true) movement.body} in
  expect_error "E_SHARED_SYNC" (Verifier.verify_kernel missing_barrier);
  let bad_layout=Layout.Register {
    size_per_thread=[1]; threads_per_warp=[32]; warps_per_cta=[1]; order=[1] } in
  let malformed={movement with body=[Load_tensor({from_global with layout=Some bad_layout},input)]} in
  expect_error "E_LAYOUT_INVALID" (Verifier.verify_kernel malformed);
  let global_with_register_layout={movement with
    args=[{name="input";value={input with layout=Some reg_layout}};{name="output";value=output}];body=[]} in
  expect_error "E_MEMREF_LAYOUT" (Verifier.verify_kernel global_with_register_layout);
  let movement_ptx=Ptx.emit movement in
  List.iter (fun part -> if not (contains movement_ptx part) then
    failwith ("shared movement PTX missing " ^ part))
    [".shared .align 16 .b8 __shared_62[128];";
     "ld.global.f32";"st.shared.f32";"bar.sync 0;";"ld.shared.f32";"st.global.f32"];
  print_endline "verifier semantic tests passed"
