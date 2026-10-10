(* Emit PTX for the OxCaml matmul kernel.

   Preferred (CTA width comes from [@@gpu.threads] on the kernel):
     tools/compile_oxcaml_kernels.sh OUT
     emit_matmul_ptx.exe --gpu OUT/matmul_tiled.gpu
     emit_matmul_ptx.exe --info OUT/matmul_tiled.gpu
     emit_matmul_ptx.exe --print-choose M N K
     emit_matmul_ptx.exe --print-config bm=128,bn=256,bk=32,stages=3,producers=32

   The tile catalog (matmul_config.ml, a module of this executable) only names
   shapes and maps them to the kernel bindings; it builds no IR:
     emit_matmul_ptx.exe --list-catalog
*)

let read_file path =
  let ch = open_in path in
  Fun.protect ~finally:(fun () -> close_in ch) (fun () ->
    really_input_string ch (in_channel_length ch))

let compile_gpu path =
  let source = Oxcaml_frontend.import (read_file path) in
  let kernel = Kernel_frontend.lower source in
  match kernel.Ir.threads_per_cta with
  | None ->
      prerr_endline "OxCaml kernel has no [@@gpu.threads N] attribute";
      exit 2
  | Some threads -> (threads, Compiler.compile kernel)

let () =
  let args = Array.to_list Sys.argv |> List.tl in
  match args with
  | ["--gpu"; path] ->
      let threads, compiled = compile_gpu path in
      Printf.eprintf "emit_matmul (OxCaml): threads=%d dynamic_smem=%d\n%!"
        threads compiled.Compiler.dynamic_shared_bytes;
      print_string compiled.Compiler.ptx
  (* Launch facts a caller cannot read off the PTX: CTA width and the dynamic
     shared memory the launch has to request. *)
  | ["--info"; path] ->
      let threads, compiled = compile_gpu path in
      Printf.printf "%d %d %d\n" threads
        compiled.Compiler.dynamic_shared_bytes compiled.Compiler.shared_bytes
  | ["--print-choose"; m; n; k] ->
      let config =
        Matmul_config.choose_config
          ~m:(int_of_string m) ~n:(int_of_string n) ~k:(int_of_string k) ()
      in
      Printf.printf "%s %d %d %d %d\n"
        (Matmul_config.kernel_binding config)
        config.bm config.bn config.bk
        (Matmul_config.threads config)
  | ["--print-config"; spec] ->
      let config = Matmul_config.parse_config spec in
      Printf.printf "%s %d %d %d %d\n"
        (Matmul_config.kernel_binding config)
        config.bm config.bn config.bk
        (Matmul_config.threads config)
  | ["--list-catalog"] ->
      List.iter
        (fun c -> print_endline (Matmul_config.string_of_config c))
        Matmul_config.tile_catalog
  | _ ->
      prerr_endline
        "usage: emit_matmul_ptx.exe --gpu FILE.gpu\n       \
         emit_matmul_ptx.exe --info FILE.gpu\n       \
         emit_matmul_ptx.exe --print-choose M N K\n       \
         emit_matmul_ptx.exe --print-config KEY=VAL,...\n       \
         emit_matmul_ptx.exe --list-catalog";
      exit 2
