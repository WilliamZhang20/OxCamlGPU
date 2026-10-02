open Kernel_ast
open Gpu_type
open Gpu_mode
let slot ty = {loc=Source_span.synthetic;ty;ownership=Aliased;locality=Global;domain_portability=Domain_nonportable;gpu_boundary=Boundary_unspecified;permission=Read_only}
let loc={Source_span.file="a\tquoted\"path.ml";start_line=2;start_col=3;end_line=2;end_col=14}
let node op=instruction ~loc op
let kernel body={name="frontend";args=[];result=slot Unit;body}
let parse_fails source = try ignore(Gpu_metadata.parse source);failwith "invalid metadata accepted" with Gpu_metadata.Parse_error _->()
let lower_fails k = try ignore(Kernel_frontend.lower k);failwith "invalid frontend accepted" with Source_span.Error _|Verifier.Invalid_kernel _->()
let () =
  let k=kernel [node(Const_bool(0,true));node(If(Some(3,I32),Value 0,
    {body=[node(Const_i32(1,2))];yield=Some(Value 1)},
    {body=[node(Const_i32(2,3))];yield=Some(Value 2)}))] in
  let encoded=Gpu_metadata.encode k in
  if Gpu_metadata.parse encoded<>k then failwith "typed source codec failed to round-trip";
  ignore(Kernel_frontend.lower k);
  parse_fails "format\t2\n";
  parse_fails "format\t3\nkernel\tx\narg\t0\tscalar_i32\tunique\taliased\tnonportable\tread\nresult\tunit\taliased\tglobal\tnonportable\tread\n";
  parse_fails (encoded ^ "end\n");
  lower_fails (kernel [node(Const_i32(0,1));node(Const_i32(0,2))]);
  lower_fails (kernel [node(Const_i32(-1,1))]);
  lower_fails (kernel [node(Add_i32(0,Value 0,Value 0))]);
  lower_fails (kernel [node(Add_i32(0,Arg 0,Arg 0))]);
  lower_fails (kernel (k.body @ [node(Add_i32(4,Value 1,Value 3))]));
  print_endline "typed frontend codec, locations, and invalid-reference checks passed"
