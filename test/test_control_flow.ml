open Gpu_type
open Ir
let ok k = Verifier.verify_exn k
let rejects code k = match Verifier.verify_kernel k with
  | Error es when List.exists (fun e->e.Verifier.code=code) es -> ()
  | _ -> failwith ("expected " ^ code)
let region ?yield body = {Ir.body;yield}
let has text fragment =
  let rec loop n = n+String.length fragment<=String.length text &&
    (String.sub text n (String.length fragment)=fragment || loop(n+1)) in loop 0
let () =
  let p=make_value ~permission:Gpu_mode.Read_write 0 (Ptr(F32,Global)) in
  let index=make_value 1 I32 and limit=make_value 2 I32 and cond=make_value 3 Bool in
  let a=make_value 4 F32 and b=make_value 5 F32 and selected=make_value 6 F32 in
  let ptr=make_value ~permission:Gpu_mode.Read_write 7 (Ptr(F32,Global)) in
  let prefix=[Thread_idx_x index;Const_i32(limit,16);Compare(cond,Lt,index,limit)] in
  let branch=If(Some selected,cond,region ~yield:a [Const_f32(a,1.)],region ~yield:b [Const_f32(b,2.)]) in
  let kernel={name="branch_ir";args=[{name="output";value=p}];body=prefix @ [branch;Gep_f32(ptr,p,index);Store_f32(ptr,selected)];threads_per_cta=None} in
  ok kernel;
  if Array.length Sys.argv=2 && Sys.argv.(1)="--ptx" then print_string(Compiler.compile_ptx kernel) else begin
  let emitted=Compiler.compile_ptx kernel in
  if not(has emitted "bra L" && has emitted "mov.f32 %f6") then failwith "missing target branch/join";
  if has emitted "bra.uni" then failwith "tid < 16 must stay a divergent branch";
  rejects "E_UNDEFINED" {kernel with body=prefix@[branch;Store_f32(p,a)]};
  rejects "E_YIELD_TYPE" {kernel with body=prefix@[If(Some selected,cond,region [Const_f32(a,1.)],region ~yield:b [Const_f32(b,2.)])]};
  rejects "E_UNDEFINED" {kernel with body=prefix@[If(Some selected,cond,region ~yield:selected [],region ~yield:b [Const_f32(b,2.)])]};
  rejects "E_REDEFINITION" {kernel with body=prefix@[If(Some selected,cond,region ~yield:a [Const_f32(a,1.)],region ~yield:a [Const_f32(a,2.)])]};
  rejects "E_CONDITION_TYPE" {kernel with body=prefix@[If(None,index,region [],region [])]};
  rejects "E_REGION_RETURN" {kernel with body=prefix@[If(None,cond,region [Return None],region [])]};
  let lane_value=make_value 8 F32 and reduced=make_value 9 F32 in
  rejects "E_DIVERGENT_COLLECTIVE" {kernel with body=prefix@[Const_f32(lane_value,1.);If(None,cond,region [Warp_sum_f32(reduced,lane_value)],region [])]};
  let yes=make_value 10 Bool in
  rejects "E_DIVERGENT_COLLECTIVE" {kernel with body=prefix@[Const_bool(yes,true);If(None,cond,region [If(None,yes,region [Barrier Execution.Cta],region [])],region [])]};
  let false_=make_value 11 Bool and choice=make_value 12 Bool in
  rejects "E_DIVERGENT_COLLECTIVE" {kernel with body=prefix@[
    If(Some choice,cond,region ~yield:yes [Const_bool(yes,true)],region ~yield:false_ [Const_bool(false_,false)]);
    If(None,choice,region [Barrier Execution.Cta],region [])]};
  ok {kernel with body=[Const_bool(yes,true);If(None,yes,region [Barrier Execution.Cta],region [])]};
  ok {kernel with body=prefix@[branch;Warp_sum_f32(reduced,selected);Store_f32(p,reduced)]};
  let remap_layout_a=Layout.Mapped_register{
    elements_per_lane=[1];lanes_per_subgroup=[32];subgroups_per_cta=[1];
    lane_mapping=Layout.Blocked;lane_order=[0];register_order=[0];subgroup_order=[0]} in
  let remap_layout_b=Layout.Mapped_register{
    elements_per_lane=[1];lanes_per_subgroup=[32];subgroups_per_cta=[1];
    lane_mapping=Layout.Interleaved;lane_order=[0];register_order=[0];subgroup_order=[0]} in
  let remap_buf=make_value ~permission:Gpu_mode.Read_write 39 (MemRef([Static 32],Float32,Global)) in
  let remap_src=make_value ~layout:remap_layout_a 40 (Tensor([Static 32],Float32)) in
  let remap_dst=make_value ~layout:remap_layout_b 41 (Tensor([Static 32],Float32)) in
  rejects "E_DIVERGENT_COLLECTIVE"
    {kernel with args=kernel.args@[{name="remap_buf";value=remap_buf}];
     body=prefix@[Load_tensor(remap_src,remap_buf);
       If(None,cond,region [Remap_tensor(remap_dst,remap_src)],region [])]};
  let shape=[Static 32] in
  let layout=Layout.Register{elements_per_lane=[1];lanes_per_subgroup=[32];subgroups_per_cta=[1];order=[0]} in
  let smem=make_value ~permission:Gpu_mode.Read_write ~layout:(Layout.Shared{vector_width=1;order=[0];swizzle=Layout.No_swizzle}) 20 (MemRef(shape,Float32,Shared)) in
  let src=make_value 21 (MemRef(shape,Float32,Global)) in
  let tile=make_value ~layout 22 (Tensor(shape,Float32)) and loaded=make_value ~layout 23 (Tensor(shape,Float32)) in
  let shared_kernel={kernel with args=kernel.args@[{name="src";value=src}];body=[Const_bool(yes,true);Shared_alloc smem;Load_tensor(tile,src);
    If(None,yes,region [Store_tensor(smem,tile)],region []);Barrier Execution.Cta;Load_tensor(loaded,smem)]} in
  (* SPMD If: a store in either arm initializes the tile for the CTA. *)
  ok shared_kernel;
  rejects "E_SHARED_UNINITIALIZED"
    {shared_kernel with body=[Const_bool(yes,true);Shared_alloc smem;Load_tensor(tile,src);
      If(None,yes,region [],region []);Barrier Execution.Cta;Load_tensor(loaded,smem)]};
  rejects "E_SHARED_SYNC" {shared_kernel with body=[Const_bool(yes,true);Shared_alloc smem;Load_tensor(tile,src);Store_tensor(smem,tile);
    If(None,yes,region [Barrier Execution.Cta],region []);Load_tensor(loaded,smem)]};
  let v1=make_value 30 F32 and v2=make_value 31 F32 in
  let aliasing={kernel with body=prefix@[Load_f32(v1,p);If(None,cond,region [Store_f32(p,v1)],region []);Load_f32(v2,p)]} in
  let optimized=Optimizer.optimize aliasing in
  let loads=List.filter (function Load_f32 _->true|_->false) (Ir.flatten optimized.body) in
  if List.length loads<>2 then failwith "branch store left a stale load cache";
  (* A branch optimizer cannot treat parent-defined derived pointers as their
     identical root address when its local address map starts empty. *)
  let root=make_value ~provenance:(Some 40) 40 (Ptr(F32,Global)) in
  let p0=make_value ~provenance:(Some 40) 41 (Ptr(F32,Global)) and p1=make_value ~provenance:(Some 40) 42 (Ptr(F32,Global)) in
  let i0=make_value 43 I32 and i1=make_value 44 I32 in
  let different={kernel with args=[{name="input";value=root}];body=[Const_bool(yes,true);Const_i32(i0,0);Const_i32(i1,1);
    Gep_f32(p0,root,i0);Gep_f32(p1,root,i1);If(None,yes,region [Load_f32(v1,p0);Load_f32(v2,p1)],region [])]} in
  let instructions=Ir.flatten (Optimizer.optimize different).body in
  if List.length(List.filter(function Load_f32 _->true|_->false) instructions)<>2 then failwith "distinct inherited pointers were merged";
  let forged={selected with domain_portability=Gpu_mode.Domain_portable} in
  rejects "E_YIELD_FACTS" {kernel with body=prefix@[If(Some forged,cond,region ~yield:a [Const_f32(a,1.)],region ~yield:b [Const_f32(b,2.)])]};
  (* 256 is warp-aligned, so the consumer split is subgroup-uniform.
     A warp sum inside it is legal; a CTA barrier is not. Elect is one lane. *)
  let wide_limit=make_value 80 I32 and wide=make_value 81 Bool and elect=make_value 82 Bool in
  let aligned cmp cond =
    [Thread_idx_x index;Const_i32(wide_limit,256);Compare(cond,cmp,index,wide_limit)] in
  ok {kernel with body=aligned Lt wide@[Const_f32(lane_value,1.);If(None,wide,region [Warp_sum_f32(reduced,lane_value)],region [])]};
  rejects "E_DIVERGENT_COLLECTIVE" {kernel with body=aligned Lt wide@[If(None,wide,region [Barrier Execution.Cta],region [])]};
  rejects "E_DIVERGENT_COLLECTIVE" {kernel with body=aligned Eq elect@[Const_f32(lane_value,1.);If(None,elect,region [Warp_sum_f32(reduced,lane_value)],region [])]};
  let shifted=make_value 84 I32 and one=make_value 85 I32 and shifted_cond=make_value 86 Bool in
  rejects "E_DIVERGENT_COLLECTIVE" {kernel with body=[
    Thread_idx_x index;Const_i32(one,1);Const_i32(wide_limit,256);Add_i32(shifted,index,one);
    Const_f32(lane_value,1.);Compare(shifted_cond,Lt,shifted,wide_limit);
    If(None,shifted_cond,region [Warp_sum_f32(reduced,lane_value)],region [])]};
  let ptx_if cmp cond =
    Compiler.compile_ptx {kernel with body=aligned cmp cond@[If(None,cond,region [Const_i32(one,1)],region [])]} in
  let wide_ptx=ptx_if Lt wide in
  if not (has wide_ptx "bra.uni") || has wide_ptx "bra L" then failwith "tid < 256 must lower to bra.uni";
  let ge=make_value 87 Bool in
  let ge_ptx=ptx_if Ge ge in
  if not (has ge_ptx "bra.uni") || has ge_ptx "bra L" then failwith "tid >= 256 must lower to bra.uni";
  let elect_ptx=ptx_if Eq elect in
  if has elect_ptx "bra.uni" || not (has elect_ptx "bra L") then failwith "tid = 256 must stay divergent";
  let karg=make_value 90 I32 and ind=make_value 91 I32 and zero=make_value 92 I32 and is0=make_value 93 Bool in
  let loop_ptx=Compiler.compile_ptx {kernel with args=kernel.args@[{name="k";value=karg}];body=[
    Const_i32(zero,0);Add_i32(ind,zero,zero);
    For(ind,karg,1,[Compare(is0,Eq,ind,zero);If(None,is0,region [Const_i32(one,1)],region [])])]} in
  if has loop_ptx "bra L" then failwith "uniform loop induction compare must be bra.uni";
  print_endline "structured branches, scope, uniformity, shared joins, and optimizer checks passed"
  end
