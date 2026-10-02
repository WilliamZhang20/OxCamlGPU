open Ir
(* Increasing numbers mean weaker agreement. Participation carries the weakest
   agreement of every enclosing selector, independently of the current value. *)
type agreement = Grid | Cta | Subgroup | Varying
let rank = function Grid->0 | Cta->1 | Subgroup->2 | Varying->3
let join a b = if rank a > rank b then a else b
let check (kernel : kernel) =
  let values=Hashtbl.create 32 and errors=ref [] in
  List.iter (fun a->Hashtbl.add values a.value.id Grid) kernel.args;
  let get v=Option.value (Hashtbl.find_opt values v.id) ~default:Varying in
  let error i text =
    let loc=match operands i with v::_->v.loc | []->Source_span.synthetic in
    errors := ("E_DIVERGENT_COLLECTIVE",Source_span.to_string loc ^ ": " ^ text)::!errors in
  let rec body participation = function
    | [] -> ()
    | Return _ :: _ -> ()
    | i::rest ->
        (match i with
         | If(dst,c,a,b) ->
             let active=join participation (get c) in
             body active a.body; body active b.body;
             Option.iter (fun d ->
               let yield r=Option.fold ~none:Grid ~some:get r.yield in
               Hashtbl.replace values d.id (join active (join (yield a) (yield b)))) dst
         | _ ->
             (match i with
              | Warp_sum_f32 _ | Reduce_sum_f32 _ when rank participation > rank Subgroup -> error i "full subgroup reduction inside divergent control flow"
              | Barrier Execution.Cta when rank participation > rank Cta -> error i "CTA barrier requires CTA-uniform participation"
              | _ -> ());
             let fact=match i with
               | Thread_idx_x _ | Global_idx_x _ | Load_f32 _ | Load_f32_masked _ | Load_tensor _ -> Varying
               | Warp_sum_f32 _ | Reduce_sum_f32 _ -> Subgroup
               | _ -> List.fold_left (fun f v->join f (get v)) Grid (operands i) in
             List.iter (fun v->Hashtbl.replace values v.id (join participation fact)) (results i));
        body participation rest in
  body Grid kernel.body;List.rev !errors
