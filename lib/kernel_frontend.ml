open Gpu_type
open Mode
open Ir

let lower (source : Kernel_ast.t) (signature : Oxcaml_frontend.signature) =
  if List.length source.args <> List.length signature.args then invalid_arg "source/signature arity mismatch";
  let next = ref (List.length signature.args) in
  let fresh ?(loc=Mode.Local) ty =
    let id = !next in incr next;
    make_value ~ownership:Aliased ~locality:loc id ty in
  let body = ref [] in
  let emit instruction = body := instruction :: !body in
  let args = List.map2 (fun source_kind arg ->
    let expected = match source_kind with
      | Kernel_ast.Buffer F32 -> MemRef ([Dynamic], Float32, Gpu_type.Global)
      | Kernel_ast.Scalar F32 -> F32
      | _ -> invalid_arg "unsupported source argument type" in
    if arg.value.ty <> expected then invalid_arg ("signature type mismatch for " ^ arg.name);
    arg) source.args signature.args in
  let rec lower_expr = function
    | Kernel_ast.Arg i -> (List.nth args i).value
    | Kernel_ast.Thread_idx_x -> let v=fresh I32 in emit (Thread_idx_x v); v
    | Kernel_ast.I32_const n -> let v=fresh I32 in emit (Const_i32(v,n)); v
    | Kernel_ast.F32_const n -> let v=fresh F32 in emit (Const_f32(v,n)); v
    | Kernel_ast.Load_f32 (arg_index, index_expr) ->
        let base=(List.nth args arg_index).value in
        let ix=lower_expr index_expr in
        let ptr = make_value ~ownership:Aliased ~locality:Mode.Local
            ~domain_portability:base.domain_portability ~gpu_boundary:base.gpu_boundary
            ~permission:base.permission ~provenance:base.provenance
            !next (Ptr(F32,Gpu_type.Global)) in
        incr next;
        emit (Gep_f32(ptr,base,ix));
        let value=fresh F32 in emit (Load_f32(value,ptr)); value
    | Kernel_ast.Add_f32 (a,b) -> let a=lower_expr a and b=lower_expr b in let v=fresh F32 in emit(Add_f32(v,a,b));v
    | Kernel_ast.Mul_f32 (a,b) -> let a=lower_expr a and b=lower_expr b in let v=fresh F32 in emit(Mul_f32(v,a,b));v
    | Kernel_ast.Warp_sum_f32 value -> let value=lower_expr value in let v=fresh F32 in emit(Warp_reduce_sum_f32(v,value));v in
  List.iter (function
    | Kernel_ast.Store_f32 (arg_index,index_expr,value_expr) ->
        let base=(List.nth args arg_index).value in let ix=lower_expr index_expr in
        let ptr = make_value ~ownership:Aliased ~locality:Mode.Local
            ~domain_portability:base.domain_portability ~gpu_boundary:base.gpu_boundary
            ~permission:base.permission ~provenance:base.provenance
            !next (Ptr(F32,Gpu_type.Global)) in
        incr next; emit(Gep_f32(ptr,base,ix));
        let value=lower_expr value_expr in emit(Store_f32(ptr,value))
    | Kernel_ast.Store_lane0_f32 (arg_index,value_expr) ->
        let base=(List.nth args arg_index).value in
        let value=lower_expr value_expr in emit(Store_f32_lane0(base,value))) source.body;
  { name=signature.name; args; body=List.rev !body }
