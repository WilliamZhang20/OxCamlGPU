let saxpy x y a =
  let open Kernel_source.F32 in
  let i = Kernel_source.Gpu.thread_idx_x () in
  Kernel_source.Gpu.store y i
    (a *. Kernel_source.Gpu.load x i +. Kernel_source.Gpu.load y i)

let source = Kernel_source.compile3_bbf ~name:"saxpy" saxpy
let signature = Oxcaml_frontend.parse_file "frontend/saxpy.mli"
let () = print_string (Ptx.emit (Kernel_frontend.lower source signature))
