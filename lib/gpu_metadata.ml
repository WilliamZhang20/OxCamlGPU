(* Shared, compiler-independent codec. The adapter and host compile these same
   sources with their respective compilers; no compiler artifacts are shared. *)
open Gpu_type
open Gpu_mode
open Kernel_ast
exception Parse_error of string
let fail s = raise (Parse_error s)
let integer s = try int_of_string s with Failure _ -> fail ("invalid integer " ^ s)
let id s = let n=integer s in if n < 0 then fail "negative value/argument id"; n
let atom = function Arg n -> "arg:" ^ string_of_int n | Value n -> "val:" ^ string_of_int n
let parse_atom s = match String.split_on_char ':' s with
  | ["arg";n] -> Arg(id n) | ["val";n] -> Value(id n) | _ -> fail "invalid value reference"
let memory = function Buffer_arg n -> "buf:" ^ string_of_int n | Pointer a -> atom a
let parse_memory s = match String.split_on_char ':' s with
  | ["buf";n] -> Buffer_arg(id n) | ["val";_] -> Pointer(parse_atom s) | _ -> fail "invalid memory reference"
let ty = function I32 -> "scalar_i32" | F32 -> "scalar_f32" | Bool -> "bool" | Unit -> "unit"
  | MemRef ([Dynamic],Float32,Global) -> "buffer_f32"
  | _ -> fail "unsupported frontend type"
let parse_ty = function "scalar_i32" -> I32 | "scalar_f32" -> F32 | "bool" -> Bool | "unit" -> Unit
  | "buffer_f32" -> MemRef([Dynamic],Float32,Global) | _ -> fail "unsupported frontend type"
let comparison = Gpu_type.string_of_comparison
let parse_comparison = function "eq"->Eq | "ne"->Ne | "lt"->Lt | "le"->Le | "gt"->Gt | "ge"->Ge | _ -> fail "invalid comparison"
let span_fields (loc : Source_span.t) =
  [Printf.sprintf "%S" loc.file;string_of_int loc.start_line;string_of_int loc.start_col;
   string_of_int loc.end_line;string_of_int loc.end_col]
let parse_span = function
  | [f;l;c;e;d] ->
      let file=try Scanf.sscanf f "%S%!" Fun.id with _->fail "invalid source filename" in
      let loc={Source_span.file;start_line=id l;start_col=id c;end_line=id e;end_col=id d} in
      if loc.end_line<loc.start_line || (loc.end_line=loc.start_line && loc.end_col<loc.start_col) then fail "invalid source span";
      loc
  | _->fail "malformed location"
let slot s =
  if s.gpu_boundary<>Boundary_unspecified then fail "source metadata cannot assert GPU boundary portability";
  [ty s.ty;
  (match s.ownership with Unique->"unique" | Aliased->"aliased");
  (match s.locality with Local->"local" | Global->"global");
  (match s.domain_portability with Domain_portable->"portable" | Domain_nonportable->"nonportable" | Domain_portability_unspecified->"unspecified");
  (match s.permission with Read_only->"read" | Write_only->"write" | Read_write->"read_write" | Immutable->"immutable")] @ span_fields s.loc
let parse_slot = function
  | [t;o;l;p;a;file;sl;sc;el;ec] -> {loc=parse_span [file;sl;sc;el;ec];ty=parse_ty t;
      ownership=(match o with "unique"->Unique | "aliased"->Aliased | _->fail "invalid ownership");
      locality=(match l with "local"->Local | "global"->Global | _->fail "invalid locality");
      domain_portability=(match p with "portable"->Domain_portable | "nonportable"->Domain_nonportable | "unspecified"->Domain_portability_unspecified | _->fail "invalid portability");
      gpu_boundary=Boundary_unspecified;
      permission=(match a with "read"->Read_only | "write"->Write_only | "read_write"->Read_write | "immutable"->Immutable | _->fail "invalid permission")}
  | _ -> fail "missing or extra mode fields"
let encode k =
  let b=Buffer.create 1024 in
  let row xs=Buffer.add_string b (String.concat "\t" xs ^ "\n") in
  let n=string_of_int in
  let rec instructions xs = List.iter (fun {op;loc} ->
    row ("loc" :: span_fields loc);
    let fields = match op with
      | Const_i32(d,v)->["const_i32";n d;n v] | Const_f32(d,v)->["const_f32";n d;Printf.sprintf "%.17g" v]
      | Const_bool(d,v)->["const_bool";n d;string_of_bool v]
      | Thread_idx_x d->["thread_idx_x";n d] | Global_idx_x d->["global_idx_x";n d]
      | Gep_f32(d,p,i)->["gep_f32";n d;memory p;atom i]
      | Load_f32(d,p)->["load_f32";n d;atom p]
      | Load_f32_masked(d,p,i,z)->["load_f32_masked";n d;atom p;atom i;atom z]
      | Add_f32(d,a,c)->["add_f32";n d;atom a;atom c] | Mul_f32(d,a,c)->["mul_f32";n d;atom a;atom c]
      | Add_i32(d,a,c)->["add_i32";n d;atom a;atom c] | Sub_i32(d,a,c)->["sub_i32";n d;atom a;atom c] | Mul_i32(d,a,c)->["mul_i32";n d;atom a;atom c]
      | Compare(d,c,a,v)->["compare";n d;comparison c;atom a;atom v]
      | Warp_reduce_sum_f32(d,v)->["warp_reduce_sum_f32";n d;atom v]
      | Store_f32(p,v)->["store_f32";atom p;atom v]
      | Store_f32_masked(p,v,i,z)->["store_f32_masked";atom p;atom v;atom i;atom z]
      | Store_grid_leader_f32(p,v)->["store_grid_leader_f32";memory p;atom v]
      | If(dst,c,_,_)->["if";(match dst with None->"-" | Some(d,_)->n d);(match dst with None->"unit" | Some(_,t)->ty t);atom c] in
    row ("body"::fields);
    match op with If(_,_,a,c)->region a; row ["else"];region c;row ["end"] | _->()) xs
  and region r = instructions r.body; row ["yield";Option.fold ~none:"-" ~some:atom r.yield] in
  row ["format";"3"];row ["kernel";k.name];
  List.iter (fun a->row (["arg";n a.index] @ slot a.slot)) k.args;
  row ("result"::slot k.result); instructions k.body; Buffer.contents b
let parse source =
  let lines=String.split_on_char '\n' source |> List.filter ((<>) "") |> Array.of_list in
  let pos=ref 0 in
  let peek () = if !pos=Array.length lines then [] else String.split_on_char '\t' lines.(!pos) in
  let take () = let r=peek() in incr pos; r in
  let expect r = if take()<>r then fail "malformed region/header" in
  expect ["format";"3"];
  let name=match take() with ["kernel";s] when s<>""->s | _->fail "missing kernel" in
  let rec args n = match peek() with
    | "arg"::i::s -> ignore(take()); if id i<>n then fail "argument ids must be contiguous";
        let slot=parse_slot s in {index=n;slot}::args(n+1)
    | _->[] in
  let args=args 0 in
  let result=match take() with "result"::s->parse_slot s | _->fail "missing result" in
  let rec instructions () = match peek() with
    | "loc"::_ ->
        let loc=match take() with "loc"::fields->parse_span fields | _->fail "missing location" in
        let op=match take() with
          | ["body";"const_i32";d;v]->Const_i32(id d,integer v)
          | ["body";"const_f32";d;v]->Const_f32(id d,(try float_of_string v with _->fail "invalid float"))
          | ["body";"const_bool";d;v]->Const_bool(id d,(match v with "true"->true|"false"->false|_->fail "invalid bool"))
          | ["body";"thread_idx_x";d]->Thread_idx_x(id d) | ["body";"global_idx_x";d]->Global_idx_x(id d)
          | ["body";"gep_f32";d;p;i]->Gep_f32(id d,parse_memory p,parse_atom i)
          | ["body";"load_f32";d;p]->Load_f32(id d,parse_atom p)
          | ["body";"load_f32_masked";d;p;i;n]->Load_f32_masked(id d,parse_atom p,parse_atom i,parse_atom n)
          | ["body";"add_f32";d;a;b]->Add_f32(id d,parse_atom a,parse_atom b)
          | ["body";"mul_f32";d;a;b]->Mul_f32(id d,parse_atom a,parse_atom b)
          | ["body";"add_i32";d;a;b]->Add_i32(id d,parse_atom a,parse_atom b)
          | ["body";"sub_i32";d;a;b]->Sub_i32(id d,parse_atom a,parse_atom b)
          | ["body";"mul_i32";d;a;b]->Mul_i32(id d,parse_atom a,parse_atom b)
          | ["body";"compare";d;c;a;b]->Compare(id d,parse_comparison c,parse_atom a,parse_atom b)
          | ["body";"warp_reduce_sum_f32";d;v]->Warp_reduce_sum_f32(id d,parse_atom v)
          | ["body";"store_f32";p;v]->Store_f32(parse_atom p,parse_atom v)
          | ["body";"store_f32_masked";p;v;i;n]->Store_f32_masked(parse_atom p,parse_atom v,parse_atom i,parse_atom n)
          | ["body";"store_grid_leader_f32";p;v]->Store_grid_leader_f32(parse_memory p,parse_atom v)
          | ["body";"if";d;t;c]->
              let dst=if d="-" then (if t<>"unit" then fail "unit branch type mismatch";None) else Some(id d,parse_ty t) in
              let a=region() in expect ["else"]; let b=region() in expect ["end"];
              If(dst,parse_atom c,a,b)
          | _->fail "malformed operation" in
        {loc;op}::instructions()
    | _->[]
  and region () = let body=instructions() in
    let yield=match take() with ["yield";"-"]->None | ["yield";v]->Some(parse_atom v) | _->fail "missing yield" in
    {body;yield} in
  let body=instructions() in
  if !pos<>Array.length lines then fail "unexpected trailing metadata";
  {name;args;result;body}
