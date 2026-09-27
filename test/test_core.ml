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
  print_endline "verifier semantic tests passed"
