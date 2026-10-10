open Gpu_type
open Gpu_mode
open Ir

let tensor_layout lane_mapping = Layout.Mapped_register {
  elements_per_lane=[2]; lanes_per_subgroup=[32]; subgroups_per_cta=[1];
  lane_mapping; lane_order=[0]; register_order=[0]; subgroup_order=[0] }

let buffer id permission =
  make_value ~locality:Global ~permission id (MemRef ([Static 64], Float32, Global))

let tile id layout =
  make_value ~locality:Local ~layout id (Tensor ([Static 64], Float32))

let multiply name lane_mapping =
  let layout=tensor_layout lane_mapping in
  let a=buffer 0 Read_only and b=buffer 1 Read_only and out=buffer 2 Write_only in
  let ta=tile 3 layout and tb=tile 4 layout and product=tile 5 layout in
  { name; args=[{name="a";value=a};{name="b";value=b};{name="out";value=out}];
    body=[Load_tensor(ta,a);Load_tensor(tb,b);Mul_tensor_f32(product,ta,tb);
      Store_tensor(out,product)]; threads_per_cta=Some 32 }

let dot64 =
  let layout=tensor_layout Layout.Interleaved in
  let a=buffer 0 Read_only and b=buffer 1 Read_only
  and out=make_value ~locality:Global ~permission:Write_only 2
      (MemRef ([Static 1], Float32, Global)) in
  let ta=tile 3 layout and tb=tile 4 layout and product=tile 5 layout
  and result=make_value 6 F32 in
  { name="dot64_interleaved";
    args=[{name="a";value=a};{name="b";value=b};{name="out";value=out}];
    body=[Load_tensor(ta,a);Load_tensor(tb,b);Mul_tensor_f32(product,ta,tb);
      Reduce_sum_f32(result,product);Store_f32_grid_leader(out,result)];
    threads_per_cta=Some 32 }

let () =
  let kernel=match Sys.argv.(1) with
    | "mul64_interleaved" -> multiply "mul64_interleaved" Layout.Interleaved
    | "mul64_blocked" -> multiply "mul64_blocked" Layout.Blocked
    | "dot64_interleaved" -> dot64
    | name -> invalid_arg ("unknown kernel: " ^ name) in
  print_string (Compiler.compile_ptx kernel)
