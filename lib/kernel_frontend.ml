open Gpu_type
open Mode
open Ir

let lower (source : Kernel_ast.t) (signature : Oxcaml_frontend.signature) =
  if signature.result.ty <> Unit then invalid_arg "GPU kernel entry points must return unit";
  if List.length source.args <> List.length signature.args then invalid_arg "source/signature arity mismatch";
  let args = List.map2 (fun source_kind arg ->
    let expected = match source_kind with
      | Kernel_ast.Buffer Kernel_ast.F32 -> MemRef ([Dynamic], Float32, Gpu_type.Global)
      | Kernel_ast.Buffer Kernel_ast.I32 -> invalid_arg "i32 GPU buffers are not supported yet"
      | Kernel_ast.Scalar Kernel_ast.F32 -> F32
      | Kernel_ast.Scalar Kernel_ast.I32 -> I32 in
    if arg.value.ty <> expected then invalid_arg ("signature type mismatch for " ^ arg.name);
    arg) source.args signature.args in
  let next = List.length args in
  let definitions = Hashtbl.create 32 in
  let body = ref [] in
  let emit instruction = body := instruction :: !body in
  let source_id id = next + id in
  let define ?(ownership=Aliased) ?(locality=Local) ?(domain_portability=Domain_portability_unspecified)
      ?(gpu_boundary=Boundary_unspecified) ?(permission=Read_only) ?provenance id ty =
    let value = make_value ~ownership ~locality ~domain_portability ~gpu_boundary
        ~permission ?provenance (source_id id) ty in
    Hashtbl.replace definitions id value;
    value in
  let atom = function
    | Kernel_ast.Arg index -> (List.nth args index).value
    | Kernel_ast.Value id ->
        (match Hashtbl.find_opt definitions id with
         | Some value -> value
         | None -> invalid_arg (Printf.sprintf "undefined frontend SSA value %d" id)) in
  let memory = function
    | Kernel_ast.Buffer_arg index -> (List.nth args index).value
    | Kernel_ast.Pointer pointer -> atom pointer in
  List.iter (function
    | Kernel_ast.Const_i32 (id, n) ->
        let dst=define id I32 in emit (Const_i32 (dst,n))
    | Kernel_ast.Const_f32 (id, n) ->
        let dst=define id F32 in emit (Const_f32 (dst,n))
    | Kernel_ast.Thread_idx_x id ->
        let dst=define id I32 in emit (Thread_idx_x dst)
    | Kernel_ast.Global_idx_x id ->
        let dst=define id I32 in emit (Global_idx_x dst)
    | Kernel_ast.Gep_f32 (id, base_ref, index_ref) ->
        let base=memory base_ref and index=atom index_ref in
        let dst=define ~ownership:Aliased ~locality:Local
            ~domain_portability:base.domain_portability ~gpu_boundary:base.gpu_boundary
            ~permission:base.permission ~provenance:base.provenance id (Ptr (F32,Gpu_type.Global)) in
        emit (Gep_f32 (dst,base,index))
    | Kernel_ast.Load_f32 (id, pointer) ->
        let dst=define id F32 in emit (Load_f32 (dst,atom pointer))
    | Kernel_ast.Load_f32_masked (id, pointer, index, bound) ->
        let dst=define id F32 in
        emit (Load_f32_masked (dst,atom pointer,atom index,atom bound))
    | Kernel_ast.Add_f32 (id, a, b) ->
        let dst=define id F32 in emit (Add_f32 (dst,atom a,atom b))
    | Kernel_ast.Mul_f32 (id, a, b) ->
        let dst=define id F32 in emit (Mul_f32 (dst,atom a,atom b))
    | Kernel_ast.Warp_reduce_sum_f32 (id, src) ->
        let dst=define id F32 in emit (Warp_sum_f32 (dst,atom src))
    | Kernel_ast.Store_f32 (pointer, value) -> emit (Store_f32 (atom pointer,atom value))
    | Kernel_ast.Store_f32_masked (pointer, value, index, bound) ->
        emit (Store_f32_masked (atom pointer,atom value,atom index,atom bound))
    | Kernel_ast.Store_grid_leader_f32 (pointer, value) ->
        emit (Store_f32_grid_leader (memory pointer,atom value))) source.body;
  { name=signature.name; args; body=List.rev !body }
