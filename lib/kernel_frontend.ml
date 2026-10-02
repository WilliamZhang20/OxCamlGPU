open Gpu_type
open Gpu_mode
open Ir

let lower (source : Kernel_ast.t) =
  if source.result.ty <> Unit then invalid_arg "GPU kernel entry points must return unit";
  let args=List.mapi (fun index (arg : Kernel_ast.argument) ->
    if arg.index<>index then invalid_arg "noncontiguous argument ids";
    let s=arg.slot in
    let provenance=match s.ty with Ptr _|MemRef _->Some index|_->None in
    {name="arg" ^ string_of_int index;value=make_value ~loc:s.loc ~ownership:s.ownership
      ~locality:s.locality ~domain_portability:s.domain_portability ~gpu_boundary:s.gpu_boundary
      ~permission:s.permission ~provenance index s.ty}) source.args in
  let used=Hashtbl.create 32 in
  let rec lower_body definitions body =
    let operand loc = function
      | Kernel_ast.Arg i -> if i<0 || i>=List.length args then Source_span.fail loc "invalid argument reference" else (List.nth args i).value
      | Kernel_ast.Value id -> match Hashtbl.find_opt definitions id with
          | Some v -> v | None -> Source_span.fail loc "undefined or out-of-scope frontend value" in
    List.map (fun ({op;loc} : Kernel_ast.instruction) ->
      let atom=operand loc in
      let define ?base id ty =
        if id<0 || id>max_int-List.length args || Hashtbl.mem used id then Source_span.fail loc "invalid or duplicate frontend SSA id";
        Hashtbl.add used id ();
        let value=match base with
          | None->make_value ~loc (List.length args+id) ty
          | Some base->make_value ~loc ~ownership:Aliased ~locality:Local
              ~domain_portability:base.domain_portability ~gpu_boundary:base.gpu_boundary
              ~permission:base.permission ~provenance:base.provenance (List.length args+id) ty in
        Hashtbl.add definitions id value;value in
      let memory = function Kernel_ast.Buffer_arg i -> atom(Kernel_ast.Arg i) | Kernel_ast.Pointer p->atom p in
      match op with
      | Kernel_ast.Const_i32(d,n)->Const_i32(define d I32,n)
      | Kernel_ast.Const_f32(d,n)->Const_f32(define d F32,n)
      | Kernel_ast.Const_bool(d,n)->Const_bool(define d Bool,n)
      | Kernel_ast.Thread_idx_x d->Thread_idx_x(define d I32)
      | Kernel_ast.Global_idx_x d->Global_idx_x(define d I32)
      | Kernel_ast.Gep_f32(d,p,i)->let p=memory p and i=atom i in Gep_f32(define ~base:p d (Ptr(F32,Global)),p,i)
      | Kernel_ast.Load_f32(d,p)->let p=atom p in Load_f32(define d F32,p)
      | Kernel_ast.Load_f32_masked(d,p,i,n)->let p=atom p and i=atom i and n=atom n in Load_f32_masked(define d F32,p,i,n)
      | Kernel_ast.Add_f32(d,a,b)->let a=atom a and b=atom b in Add_f32(define d F32,a,b)
      | Kernel_ast.Mul_f32(d,a,b)->let a=atom a and b=atom b in Mul_f32(define d F32,a,b)
      | Kernel_ast.Add_i32(d,a,b)->let a=atom a and b=atom b in Add_i32(define d I32,a,b)
      | Kernel_ast.Sub_i32(d,a,b)->let a=atom a and b=atom b in Sub_i32(define d I32,a,b)
      | Kernel_ast.Mul_i32(d,a,b)->let a=atom a and b=atom b in Mul_i32(define d I32,a,b)
      | Kernel_ast.Compare(d,c,a,b)->let a=atom a and b=atom b in Compare(define d Bool,c,a,b)
      | Kernel_ast.Warp_reduce_sum_f32(d,v)->let v=atom v in Warp_sum_f32(define d F32,v)
      | Kernel_ast.Store_f32(p,v)->Store_f32(atom p,atom v)
      | Kernel_ast.Store_f32_masked(p,v,i,n)->Store_f32_masked(atom p,atom v,atom i,atom n)
      | Kernel_ast.Store_grid_leader_f32(p,v)->Store_f32_grid_leader(memory p,atom v)
      | Kernel_ast.If(dst,c,a,b)->
          let c=atom c in
          let region (r : Kernel_ast.region) =
            let env=Hashtbl.copy definitions in
            let body=lower_body env r.body in
            let yield=Option.map (fun v-> match v with Kernel_ast.Arg _->atom v | Kernel_ast.Value i ->
              match Hashtbl.find_opt env i with Some v->v | None->Source_span.fail loc "undefined branch yield") r.yield in
            {body;yield} in
          let a=region a in let b=region b in
          let dst=Option.map (fun (d,t)->define d t) dst in If(dst,c,a,b)) body in
  let body=lower_body (Hashtbl.create 32) source.body in
  let kernel={name=source.name;args;body} in
  Verifier.verify_exn kernel; kernel

let () = Printexc.register_printer (function
  | Source_span.Error (loc, message) -> Some (Source_span.to_string loc ^ ": " ^ message)
  | _ -> None)
