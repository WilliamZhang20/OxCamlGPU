let read_file path =
  let channel = open_in path in
  Fun.protect ~finally:(fun () -> close_in channel) (fun () ->
    really_input_string channel (in_channel_length channel))

let () =
  if Array.length Sys.argv <> 5 then failwith "usage: check_compiler_metadata SAXPY_METADATA VECTOR_ADD_METADATA LITERAL_METADATA DOT_METADATA";
  let malformed = "format\t3\nkernel\tbad\narg\t0\tbuffer_f32\tunique\taliased\tnonportable\tread_write\nresult\tunit\taliased\tglobal\tnonportable\tread_write\n" in
  (match Gpu_metadata.parse malformed with
   | _ -> failwith "conflicting modes should be rejected by shared metadata parser"
   | exception Gpu_metadata.Parse_error _ -> ());
  let check path name =
    let metadata=read_file path in
    let source=Oxcaml_frontend.import metadata in
    let kernel=Kernel_frontend.lower source in
    Verifier.verify_exn kernel;
    if source.name <> name then failwith "kernel name changed during Typedtree import";
    kernel in
  let saxpy=check Sys.argv.(1) "saxpy" and vector_add=check Sys.argv.(2) "vector_add" in
  let literal=check Sys.argv.(3) "literal_probe" in
  let dot=check Sys.argv.(4) "dot_product" in
  if not (List.exists (function Ir.Global_idx_x _ -> true | _ -> false) vector_add.body) ||
     not (List.exists (function Ir.Load_f32_masked _ -> true | _ -> false) vector_add.body) ||
     not (List.exists (function Ir.Store_f32_masked _ -> true | _ -> false) vector_add.body) then
    failwith "global indexing or masked memory operations did not reach GPU IR";
  let input=(List.nth saxpy.args 0).value and output=(List.nth saxpy.args 1).value in
  if input.ty <> Gpu_type.MemRef ([Gpu_type.Dynamic], Gpu_type.Float32, Gpu_type.Global) ||
     output.ty <> Gpu_type.MemRef ([Gpu_type.Dynamic], Gpu_type.Float32, Gpu_type.Global) then
    failwith "GPU arrays were not represented as dynamic global f32 MemRefs";
  if input.ownership<>Gpu_mode.Aliased || input.permission<>Gpu_mode.Read_only ||
     output.ownership<>Gpu_mode.Unique || output.permission<>Gpu_mode.Read_write then
    failwith "OxCaml typedtree modes did not reach GPU IR";
  if input.gpu_boundary<>Gpu_mode.Boundary_portable ||
     output.gpu_boundary<>Gpu_mode.Boundary_portable then
    failwith "buffer formals did not derive GPU-boundary portability at lower";
  if (List.nth saxpy.args 2).value.gpu_boundary<>Gpu_mode.Boundary_unspecified then
    failwith "scalar formals incorrectly acquired GPU-boundary portability";
  if (List.nth vector_add.args 2).value.ownership<>Gpu_mode.Unique then
    failwith "OxCaml vector_add uniqueness did not reach GPU IR";
  if not (List.exists (function Ir.Const_f32 (_, 2.5) -> true | _ -> false) literal.body) then
    failwith "Typedtree float literal did not reach GPU IR";
  if not (List.exists (function Ir.Warp_sum_f32 _ -> true | _ -> false) dot.body) ||
     not (List.exists (function Ir.Store_f32_grid_leader _ -> true | _ -> false) dot.body) then
    failwith "OxCaml dot-product reduction did not reach verified GPU IR";
  if Gpu_type.string_of_ty (Gpu_type.Tensor ([Gpu_type.Static 128], Gpu_type.Float32)) <>
     "tensor<[128],f32>" then failwith "Tensor type shape/dtype printing changed";
  if Execution.parent Execution.Lane <> Some Execution.Subgroup ||
     Execution.parent Execution.Subgroup <> Some Execution.Cta ||
     Execution.parent Execution.Cta <> Some Execution.Grid ||
     Execution.parent Execution.Warpgroup <> None then
    failwith "GPU execution hierarchy is malformed";
  print_endline "OxCaml Typedtree bodies, modes, and literals lowered into verified GPU IR"
