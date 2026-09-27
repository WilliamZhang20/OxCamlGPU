let read_file path =
  let channel = open_in path in
  Fun.protect ~finally:(fun () -> close_in channel) (fun () ->
    really_input_string channel (in_channel_length channel))

let () =
  if Array.length Sys.argv <> 5 then failwith "usage: check_compiler_metadata SAXPY_METADATA VECTOR_ADD_METADATA LITERAL_METADATA DOT_METADATA";
  let malformed = "format\t1\nkernel\tbad\narg\t0\tbuffer_f32\tunique\taliased\tnonportable\tread_write\nresult\tunit\taliased\tglobal\tnonportable\tread_write\n" in
  (match Gpu_metadata.parse malformed with
   | _ -> failwith "conflicting modes should be rejected by shared metadata parser"
   | exception Gpu_metadata.Parse_error _ -> ());
  let check path name =
    let metadata=read_file path in
    let signature=Oxcaml_frontend.parse_typedtree_metadata metadata in
    let source=Kernel_ast.of_compiler_metadata metadata in
    let kernel=Kernel_frontend.lower source signature in
    Verifier.verify_exn kernel;
    if signature.name <> name || source.name <> name then failwith "kernel name changed during Typedtree import";
    kernel in
  let saxpy=check Sys.argv.(1) "saxpy" and vector_add=check Sys.argv.(2) "vector_add" in
  let literal=check Sys.argv.(3) "literal_probe" in
  let dot=check Sys.argv.(4) "dot_product" in
  let input=(List.nth saxpy.args 0).value and output=(List.nth saxpy.args 1).value in
  if input.ty <> Gpu_type.MemRef ([Gpu_type.Dynamic], Gpu_type.Float32, Gpu_type.Global) ||
     output.ty <> Gpu_type.MemRef ([Gpu_type.Dynamic], Gpu_type.Float32, Gpu_type.Global) then
    failwith "GPU arrays were not represented as dynamic global f32 MemRefs";
  if input.ownership<>Mode.Aliased || input.permission<>Mode.Read_only ||
     output.ownership<>Mode.Unique || output.permission<>Mode.Read_write then
    failwith "OxCaml typedtree modes did not reach GPU IR";
  if (List.nth vector_add.args 2).value.ownership<>Mode.Unique then
    failwith "OxCaml vector_add uniqueness did not reach GPU IR";
  if not (List.exists (function Ir.Const_f32 (_, 2.5) -> true | _ -> false) literal.body) then
    failwith "Typedtree float literal did not reach GPU IR";
  if not (List.exists (function Ir.Warp_reduce_sum_f32 _ -> true | _ -> false) dot.body) ||
     not (List.exists (function Ir.Store_f32_lane0 _ -> true | _ -> false) dot.body) then
    failwith "OxCaml dot-product reduction did not reach verified GPU IR";
  if Gpu_type.string_of_ty (Gpu_type.Tensor ([Gpu_type.Static 128], Gpu_type.Float32)) <>
     "tensor<[128],f32>" then failwith "Tensor type shape/dtype printing changed";
  if Gpu_type.parent_level Gpu_type.Lane <> Some Gpu_type.Warp ||
     Gpu_type.parent_level Gpu_type.Warp <> Some Gpu_type.Warpgroup ||
     Gpu_type.parent_level Gpu_type.Warpgroup <> Some Gpu_type.Cta ||
     Gpu_type.parent_level Gpu_type.Cta <> Some Gpu_type.Grid then
    failwith "GPU execution hierarchy is malformed";
  print_endline "OxCaml Typedtree bodies, modes, and literals lowered into verified GPU IR"
