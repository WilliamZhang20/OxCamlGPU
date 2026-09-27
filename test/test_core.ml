open Gpu_type
open Gpu_ir

let expect_ok = function Ok () -> () | Error es -> failwith (String.concat "; " (List.map (fun e -> e.Verifier.code ^ ": " ^ e.message) es))
let expect_error code = function
  | Error es when List.exists (fun e -> e.Verifier.code = code) es -> ()
  | Error es -> failwith ("expected " ^ code ^ ", got " ^ String.concat "," (List.map (fun e -> e.Verifier.code) es))
  | Ok () -> failwith ("expected error " ^ code)
let actual ?(ownership=Aliased) ?(permission=Read_only) buffer_id =
  { buffer_id; actual_ownership=ownership; actual_permission=permission }

let () =
  expect_ok (Verifier.verify_kernel Kernel.vector_add);
  expect_ok (Verifier.verify_kernel Kernel.saxpy);
  expect_ok (Verifier.check_call Kernel.vector_add [
    actual 10; actual 11; actual ~ownership:Unique ~permission:Read_write 12]);
  expect_error "E_UNIQUE_ALIAS" (Verifier.check_call Kernel.vector_add [
    actual 10; actual 11; actual ~ownership:Unique ~permission:Read_write 11]);
  expect_error "E_UNIQUE_REQUIRED" (Verifier.check_call Kernel.saxpy [
    actual 10; actual 11; actual 20]);
  expect_error "E_ACCESS_REQUIRED" (Verifier.check_call Kernel.saxpy [
    actual 10; actual ~ownership:Unique ~permission:Read_only 11;
    actual ~permission:Read_write 20]);
  let local = Gpu_ir.make_value ~locality:Local 40 (Ptr (F32, Global)) in
  let returning = { Kernel.vector_add with name="bad_escape"; body=[Return (Some local)] } in
  expect_error "E_LOCAL_ESCAPE" (Verifier.verify_kernel returning);
  let ro = Gpu_ir.make_value ~permission:Read_only 50 (Ptr (F32, Global)) in
  let x = Gpu_ir.make_value 51 F32 in
  let readonly_store = { name="bad_store"; args=[{name="p";value=ro}]; body=[Const_f32 (x, 1.); Store_f32 (ro,x)] } in
  expect_error "E_READONLY_STORE" (Verifier.verify_kernel readonly_store);
  let wo = Gpu_ir.make_value ~permission:Write_only 52 (Ptr (F32, Global)) in
  let loaded = Gpu_ir.make_value 53 F32 in
  let writeonly_load = { name="bad_load"; args=[{name="p";value=wo}]; body=[Load_f32 (loaded,wo)] } in
  expect_error "E_UNREADABLE_LOAD" (Verifier.verify_kernel writeonly_load);
  let ptx = Ptx.emit Kernel.vector_add in
  let contains text needle =
    let n = String.length needle in
    let rec loop i = i + n <= String.length text &&
      (String.sub text i n = needle || loop (i + 1)) in
    loop 0 in
  List.iter (fun s -> if not (contains ptx s) then failwith ("vector_add PTX missing " ^ s))
    [".target sm_90"; ".param .u64 %arg0"; "mov.u32 %r3, %tid.x"; "ld.global.f32"; "add.f32"; "st.global.f32"];
  let saxpy_ptx = Ptx.emit Kernel.saxpy in
  List.iter (fun s -> if not (contains saxpy_ptx s) then failwith ("saxpy PTX missing " ^ s))
    [".param .f32 %arg2"; "mul.f32"; "add.f32"; "st.global.f32"];
  let imported = Oxcaml_frontend.parse_string
      "type f32\ntype 'a gpu_array\nval saxpy :\n f32 gpu_array @ aliased read ->\n f32 gpu_array @ unique read_write ->\n f32 ->\n unit" in
  if imported.name <> "saxpy" || imported.result.ty <> Unit || List.length imported.args <> 3 then
    failwith "OxCaml signature import shape mismatch";
  let input, output, alpha = match imported.args with [x;y;a] -> x.value,y.value,a.value | _ -> assert false in
  if input.ownership <> Aliased || input.permission <> Read_only || output.ownership <> Unique ||
     output.permission <> Read_write || alpha.ty <> F32 then
    failwith "OxCaml ownership modes did not reach GPU IR";
  let modal = Oxcaml_frontend.parse_string
      "val f : f32 gpu_array @ local unique portable read -> unit" in
  let imported_arg = List.hd modal.args in
  if imported_arg.value.locality <> Local || imported_arg.value.ownership <> Unique ||
     imported_arg.value.domain_portability <> Domain_portable ||
     imported_arg.value.gpu_boundary <> Boundary_unspecified ||
     imported_arg.value.permission <> Read_only then
    failwith "OxCaml locality or portability mode was lost";
  let result_modes = Oxcaml_frontend.parse_string
      "val returns_local : f32 gpu_array @ unique read -> f32 gpu_array @ local aliased read" in
  if result_modes.result.locality <> Local || result_modes.result.permission <> Read_only ||
     result_modes.result.ownership <> Aliased then
    failwith "OxCaml result modes were dropped";
  print_endline "semantic core tests passed"
