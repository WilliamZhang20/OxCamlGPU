(* Walk the version-specific Typedtree and construct the shared Kernel_ast. *)
open Typedtree
module K = Kernel_ast
module G = Gpu_type
module T = Typedtree_support
open T
type binding = Scalar of K.atom * (int64 * int64) option | Memory of K.memory | Void
type state = { mutable next : int; mutable body : K.instruction list }
let fresh s = let n=s.next in s.next<-n+1;n
let emit s exp op = s.body <- K.instruction ~loc:(span exp.exp_loc) op :: s.body
let full_range=Some(-2147483648L,2147483647L)
let get env exp path = match path with
  | Path.Pident id -> (match List.find_opt (fun (key,_)->Ident.same key id) env with Some(_,v)->v | None->fail exp.exp_loc "unbound GPU value")
  | _->fail exp.exp_loc "unsupported nonlocal value"
let scalar loc = function Scalar(a,r)->a,r | _->fail loc "expected scalar"
let memory loc = function Memory m->m | _->fail loc "expected GPU buffer"
let unit loc = function Void->() | _->fail loc "expected unit"
let checked loc (lo,hi) = if lo < -2147483648L || hi > 2147483647L then fail loc "native int arithmetic may overflow the supported i32 range; use explicit Int32 arithmetic" else Some(lo,hi)
let args loc xs = List.map (function Nolabel,Typedtree.Arg(e,_)->e | _->fail loc "only positional fully applied calls are supported") xs
let rec expression s env exp =
  let const op range = let d=fresh s in emit s exp (op d); Scalar(K.Value d,range) in
  let result op = const op None in
  match exp.exp_desc with
  | Texp_constant(Const_int n) -> ignore(checked exp.exp_loc (Int64.of_int n,Int64.of_int n));const (fun d->K.Const_i32(d,n)) (Some(Int64.of_int n,Int64.of_int n))
  | Texp_constant(Const_int32 n) -> const (fun d->K.Const_i32(d,Int32.to_int n)) (Some(Int64.of_int32 n,Int64.of_int32 n))
  | Texp_constant(Const_float n) -> result(fun d->K.Const_f32(d,float_of_string n))
  | Texp_ident {path;_}->get env exp path
  | Texp_open(_,e)->expression s env e
  | Texp_construct(_,c,_,[],_) ->
      (match kind exp.exp_env exp.exp_loc exp.exp_type,c.cstr_name with
       | Unit,"()"->Void | Boolean,"true"->result(fun d->K.Const_bool(d,true)) | Boolean,"false"->result(fun d->K.Const_bool(d,false))
       | _->fail exp.exp_loc "unsupported constructor")
  | Texp_let(Nonrecursive,[binding],body) ->
      let value=expression s env binding.vb_expr in
      (match binding.vb_pat.pat_desc with
       | Tpat_var {id;_}->expression s ((id,value)::env) body
       | Tpat_any->expression s env body
       | _->fail binding.vb_pat.pat_loc "only simple kernel let bindings are supported")
  | Texp_sequence(a,_,b)->unit a.exp_loc (expression s env a);expression s env b
  | Texp_ifthenelse(c,a,b)->
      let c,_=scalar c.exp_loc (expression s env c) in
      conditional s exp c (fun()->expression s env a) (fun()->Option.fold ~none:Void ~some:(expression s env) b)
  | Texp_apply(callee,xs,_,_,_,_) ->
      let op=primitive callee and xs=args exp.exp_loc xs in
      (match op,xs with
       | ("&&"|"||"),[a;b] ->
           let a,_=scalar a.exp_loc(expression s env a) in
           let literal value () = result(fun d->K.Const_bool(d,value)) in
           if op="&&" then conditional s exp a (fun()->expression s env b) (literal false)
           else conditional s exp a (literal true) (fun()->expression s env b)
       | "not",[a] ->
           let a,_=scalar a.exp_loc(expression s env a) in
           conditional s exp a (fun()->result(fun d->K.Const_bool(d,false))) (fun()->result(fun d->K.Const_bool(d,true)))
       | ("thread_idx_x"|"global_idx_x"),[u] ->
           unit u.exp_loc (expression s env u);
           const (fun d->if op="thread_idx_x" then K.Thread_idx_x d else K.Global_idx_x d) (Some(0L,2147483647L))
       | ("load"|"load_masked"),buffer::index::tail ->
           let base=memory buffer.exp_loc(expression s env buffer) in
           let ix,_=scalar index.exp_loc(expression s env index) in
           let bound=match op,tail with "load",[]->None | "load_masked",[n]->Some(fst(scalar n.exp_loc(expression s env n))) | _->fail exp.exp_loc "invalid load arity" in
           let p=fresh s in emit s exp (K.Gep_f32(p,base,ix));
           result(fun d->match bound with None->K.Load_f32(d,K.Value p) | Some n->K.Load_f32_masked(d,K.Value p,ix,n))
       | ("store"|"store_masked"),buffer::index::tail ->
           let base=memory buffer.exp_loc(expression s env buffer) in
           let ix,_=scalar index.exp_loc(expression s env index) in
           let bound,value=match op,tail with "store",[v]->None,v | "store_masked",[n;v]->Some(fst(scalar n.exp_loc(expression s env n))),v | _->fail exp.exp_loc "invalid store arity" in
           let p=fresh s in emit s exp (K.Gep_f32(p,base,ix));
           let v,_=scalar value.exp_loc(expression s env value) in
           emit s exp (match bound with None->K.Store_f32(K.Value p,v)|Some n->K.Store_f32_masked(K.Value p,v,ix,n));Void
       | "store_grid_leader",[buffer;v] ->
           let p=memory buffer.exp_loc(expression s env buffer) in let v,_=scalar v.exp_loc(expression s env v) in
           emit s exp (K.Store_grid_leader_f32(p,v));Void
       | "warp_sum_f32",[v]->let v,_=scalar v.exp_loc(expression s env v) in result(fun d->K.Warp_reduce_sum_f32(d,v))
       | ("i32_of_int"|"i32_to_int"),[v]->expression s env v
       | ("+."|"*."|"+"|"-"|"*"|"i32_add"|"i32_sub"|"i32_mul"|"="|"<>"|"<"|"<="|">"|">="),[a;b] ->
           let av,ar=scalar a.exp_loc(expression s env a) in
           let bv,br=scalar b.exp_loc(expression s env b) in
           (match op with
            | "+."->result(fun d->K.Add_f32(d,av,bv)) | "*."->result(fun d->K.Mul_f32(d,av,bv))
            | "+"|"-"|"*"|"i32_add"|"i32_sub"|"i32_mul" ->
                let range=if String.starts_with ~prefix:"i32_" op then full_range else
                  match ar,br with
                  | Some(al,ah),Some(bl,bh)->
                      let lo,hi=match op with
                        | "+"->Int64.add al bl,Int64.add ah bh
                        | "-"->Int64.sub al bh,Int64.sub ah bl
                        | _->let ps=List.map (fun (x,y)->Int64.mul x y) [al,bl;al,bh;ah,bl;ah,bh] in List.fold_left min Int64.max_int ps,List.fold_left max Int64.min_int ps in
                      checked exp.exp_loc (lo,hi)
                  | _->fail exp.exp_loc "cannot prove native int arithmetic fits i32; use Int32 arithmetic" in
                const (fun d-> match op with "+"|"i32_add"->K.Add_i32(d,av,bv)|"-"|"i32_sub"->K.Sub_i32(d,av,bv)|_->K.Mul_i32(d,av,bv)) range
            | _ ->
                let k=kind a.exp_env a.exp_loc a.exp_type in
                if k=Buffer || k=Unit then fail exp.exp_loc "only scalar comparisons are supported";
                let c=match op with "="->G.Eq|"<>"->G.Ne|"<"->G.Lt|"<="->G.Le|">"->G.Gt|_->G.Ge in
                if k=Boolean && c<>G.Eq && c<>G.Ne then fail exp.exp_loc "only boolean equality comparisons are supported";
                result(fun d->K.Compare(d,c,av,bv)))
       | _->fail exp.exp_loc ("unsupported GPU call/arity: " ^ op))
  | _->fail exp.exp_loc "unsupported GPU expression"
and conditional s exp condition yes no =
  let outer=s.body in
  let region f = s.body<-[]; let value=f() in
    let yield=match value with Void->None | Scalar(a,_)->Some a | Memory _->fail exp.exp_loc "buffer-valued branches are unsupported" in
    {K.body=List.rev s.body;yield},value in
  let a,av=region yes in let b,bv=region no in s.body<-outer;
  let dst,value=match av,bv with
    | Void,Void -> None,Void
    | Scalar(_,ar),Scalar(_,br)->
        let d=fresh s in
        let range=match ar,br with Some(al,ah),Some(bl,bh)->Some(min al bl,max ah bh)|_->None in
        Some(d,gpu_ty(kind exp.exp_env exp.exp_loc exp.exp_type)),Scalar(K.Value d,range)
    | _->fail exp.exp_loc "conditional arms have incompatible results" in
  emit s exp (K.If(dst,condition,a,b));value
let compile target structure =
  let binding=List.find_map (fun i->match i.str_desc with
    | Tstr_value(_,bs)->List.find_opt (fun b->match b.vb_pat.pat_desc with Tpat_var{name;_}->name.txt=target | _->false) bs | _->None) structure.str_items in
  let binding=match binding with Some b->b|None->fail Location.none "kernel binding not found" in
  let env=binding.vb_expr.exp_env and loc=binding.vb_expr.exp_loc in
  let rec signature i typ last = match Types.get_desc typ with
    | Types.Tpoly(t,_)->signature i t last
    | Types.Tarrow((_,am,rm),a,b,_)->let a={K.index=i;slot=slot env loc a am} in let rest,result=signature(i+1)b(Some rm) in a::rest,result
    | _->match last with Some m->[],slot env loc typ m|None->fail loc "kernel must be a function" in
  let args,result=signature 0 binding.vb_pat.pat_type None in
  if result.ty<>G.Unit then fail loc "GPU entry kernels must return unit";
  match binding.vb_expr.exp_desc with
  | Texp_function {params;body=Tfunction_body body;_}->
      let env=List.mapi (fun i param->
        let k=match param.fp_kind with Tparam_pat p->kind p.pat_env p.pat_loc p.pat_type|_->fail loc "optional parameters unsupported" in
        let value=match k with Buffer->Memory(K.Buffer_arg i)|Unit->fail loc "unit kernel argument unsupported"|Native_int|Int32->Scalar(K.Arg i,full_range)|_->Scalar(K.Arg i,None) in
        param.fp_param,value) params in
      let s={next=0;body=[]} in unit body.exp_loc (expression s env body);
      {K.name=target;args;result;body=List.rev s.body}
  | _->fail loc "kernel must have a simple function body"
let () =
  try
    if Array.length Sys.argv<>3 then fail Location.none "usage: export_typedtree_modes FILE.cmt NAME";
    let _,cmt=Cmt_format.read Sys.argv.(1) in
    match cmt with
    | Some cmt ->
        Load_path.init ~auto_include:Load_path.no_auto_include ~visible:cmt.cmt_loadpath.visible ~hidden:cmt.cmt_loadpath.hidden;
        (match cmt.cmt_annots with Implementation s->print_string(Gpu_metadata.encode(compile Sys.argv.(2) s))|_->fail Location.none "expected implementation")
    | None->fail Location.none "missing typedtree"
  with Source_span.Error(loc,msg)->prerr_endline(Source_span.to_string loc ^ ": " ^ msg);exit 2
