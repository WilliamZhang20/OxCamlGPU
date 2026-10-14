(* Read a .cmt, trace one kernel of it into Kernel_ast, and print the encoded
   result. Errors carry a source span and exit with status 2. *)
open Typedtree_support

let () =
  try
    if Array.length Sys.argv <> 3 then
      fail Location.none "usage: export_typedtree_modes FILE.cmt NAME";
    let _, cmt = Cmt_format.read Sys.argv.(1) in
    match cmt with
    | Some cmt ->
        Load_path.init
          ~auto_include:Load_path.no_auto_include
          ~visible:cmt.cmt_loadpath.visible
          ~hidden:cmt.cmt_loadpath.hidden;
        (match cmt.cmt_annots with
         | Implementation s ->
             print_string
               (Gpu_metadata.encode (Export_kernel.compile Sys.argv.(2) s))
         | _ -> fail Location.none "expected implementation")
    | None -> fail Location.none "missing typedtree"
  with Source_span.Error (loc, msg) ->
    prerr_endline (Source_span.to_string loc ^ ": " ^ msg);
    exit 2
