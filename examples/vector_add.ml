let vector_add x y z =
  let open Kernel_source.F32 in
  let i = Kernel_source.Gpu.thread_idx_x () in
  Kernel_source.Gpu.store z i
    (Kernel_source.Gpu.load x i +. Kernel_source.Gpu.load y i)

let source = Kernel_source.compile3_bbb ~name:"vector_add" vector_add
let signature = Oxcaml_frontend.parse_string
    "val vector_add : f32 gpu_array @ aliased read -> f32 gpu_array @ aliased read -> f32 gpu_array @ unique read_write -> unit"
let () = print_string (Ptx.emit (Kernel_frontend.lower source signature))
