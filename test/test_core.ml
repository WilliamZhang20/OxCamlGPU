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

let saxpy x y a =
  let open Kernel_source.F32 in
  let i = Kernel_source.Gpu.thread_idx_x () in
  Kernel_source.Gpu.store y i
    (a *. Kernel_source.Gpu.load x i +. Kernel_source.Gpu.load y i)
let vector_add x y z =
  let open Kernel_source.F32 in
  let i = Kernel_source.Gpu.thread_idx_x () in
  Kernel_source.Gpu.store z i (Kernel_source.Gpu.load x i +. Kernel_source.Gpu.load y i)
let saxpy_source = Kernel_source.compile3_bbf ~name:"saxpy" saxpy
let vector_source = Kernel_source.compile3_bbb ~name:"vector_add" vector_add
let saxpy_sig = Oxcaml_frontend.parse_string
  "val saxpy : f32 gpu_array @ aliased read -> f32 gpu_array @ unique read_write -> f32 -> unit"
let vector_sig = Oxcaml_frontend.parse_string
  "val vector_add : f32 gpu_array @ aliased read -> f32 gpu_array @ aliased read -> f32 gpu_array @ unique read_write -> unit"
let saxpy_ir = Kernel_frontend.lower saxpy_source saxpy_sig
let vector_ir = Kernel_frontend.lower vector_source vector_sig

let contains text needle =
  let n = String.length needle in
  let rec loop i = i + n <= String.length text &&
    (String.sub text i n = needle || loop (i + 1)) in loop 0

let () =
  expect_ok (Verifier.verify_kernel vector_ir);
  expect_ok (Verifier.verify_kernel saxpy_ir);
  expect_ok (Verifier.check_call vector_ir [actual 10;actual 11;actual ~ownership:Unique ~permission:Read_write 12]);
  expect_error "E_UNIQUE_ALIAS" (Verifier.check_call vector_ir [actual 10;actual 11;actual ~ownership:Unique ~permission:Read_write 11]);
  expect_error "E_UNIQUE_REQUIRED" (Verifier.check_call saxpy_ir [actual 10;actual 11;actual ~permission:Read_write 20]);
  let local = make_value ~locality:Mode.Local 40 (Ptr(F32,Gpu_type.Global)) in
  let returning = { vector_ir with name="bad_escape"; body=[Return(Some local)] } in
  expect_error "E_LOCAL_ESCAPE" (Verifier.verify_kernel returning);
  let ro=make_value ~permission:Read_only 50 (Ptr(F32,Gpu_type.Global)) and x=make_value 51 F32 in
  let readonly_store={name="bad_store";args=[{name="p";value=ro}];body=[Const_f32(x,1.);Store_f32(ro,x)]} in
  expect_error "E_READONLY_STORE" (Verifier.verify_kernel readonly_store);
  let ptx=Ptx.emit vector_ir in
  List.iter (fun s -> if not (contains ptx s) then failwith ("vector_add PTX missing "^s))
    [".target sm_90";".param .u64 %arg0";"mov.u32 %r3, %tid.x";"ld.global.f32";"add.f32";"st.global.f32"];
  let saxpy_ptx=Ptx.emit saxpy_ir in
  List.iter (fun s -> if not (contains saxpy_ptx s) then failwith ("saxpy PTX missing "^s))
    [".param .f32 %arg2";"mul.f32";"add.f32";"st.global.f32"];
  let input=(List.nth saxpy_ir.args 0).value and output=(List.nth saxpy_ir.args 1).value in
  if input.ownership<>Aliased || input.permission<>Read_only || output.ownership<>Unique || output.permission<>Read_write then
    failwith "OxCaml modes were not preserved through source AST lowering";
  print_endline "semantic core tests passed"
