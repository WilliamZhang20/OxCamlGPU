let read_file path =
  let ch = open_in path in
  Fun.protect ~finally:(fun () -> close_in ch) (fun () -> really_input_string ch (in_channel_length ch))

let () =
  if Array.length Sys.argv <> 2 then failwith "usage: saxpy.exe SAXPY.gpu";
  let metadata = read_file Sys.argv.(1) in
  let signature = Oxcaml_frontend.parse_typedtree_metadata metadata in
  let source = Kernel_ast.of_compiler_metadata metadata in
  print_string (Ptx.emit (Kernel_frontend.lower source signature))
