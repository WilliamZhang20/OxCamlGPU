let read_file path =
  let ch = open_in path in
  Fun.protect ~finally:(fun () -> close_in ch) (fun () -> really_input_string ch (in_channel_length ch))

let () =
  if Array.length Sys.argv <> 2 then failwith "usage: saxpy.exe SAXPY.gpu";
  let metadata = read_file Sys.argv.(1) in
  let source = Oxcaml_frontend.import metadata in
  print_string (Compiler.compile_ptx (Kernel_frontend.lower source))
