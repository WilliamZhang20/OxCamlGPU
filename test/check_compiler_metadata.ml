let read_file path =
  let channel = open_in path in
  Fun.protect ~finally:(fun () -> close_in channel) (fun () ->
    really_input_string channel (in_channel_length channel))

let () =
  if Array.length Sys.argv <> 4 then failwith "usage: check_compiler_metadata SAXPY_METADATA VECTOR_ADD_METADATA LITERAL_METADATA";
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
  let input=(List.nth saxpy.args 0).value and output=(List.nth saxpy.args 1).value in
  if input.ownership<>Mode.Aliased || input.permission<>Mode.Read_only ||
     output.ownership<>Mode.Unique || output.permission<>Mode.Read_write then
    failwith "OxCaml typedtree modes did not reach GPU IR";
  if (List.nth vector_add.args 2).value.ownership<>Mode.Unique then
    failwith "OxCaml vector_add uniqueness did not reach GPU IR";
  if not (List.exists (function Ir.Const_f32 (_, 2.5) -> true | _ -> false) literal.body) then
    failwith "Typedtree float literal did not reach GPU IR";
  print_endline "OxCaml Typedtree bodies, modes, and literals lowered into verified GPU IR"
