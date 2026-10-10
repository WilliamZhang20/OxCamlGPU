open Ir
(* Increasing numbers mean weaker agreement. Participation carries the weakest
   agreement of every enclosing selector, independently of the current value. *)
type agreement = Grid | Cta | Subgroup | Varying
let rank = function Grid -> 0 | Cta -> 1 | Subgroup -> 2 | Varying -> 3
let join a b = if rank a > rank b then a else b
let ( <=: ) a b = rank a <= rank b

let warp = 32

let rem_warp n =
  let r = n mod warp in
  if r < 0 then r + warp else r

let flip_cmp = function
  | Gpu_type.Lt -> Gpu_type.Gt
  | Gpu_type.Gt -> Gpu_type.Lt
  | Gpu_type.Le -> Gpu_type.Ge
  | Gpu_type.Ge -> Gpu_type.Le
  | Gpu_type.Eq -> Gpu_type.Eq
  | Gpu_type.Ne -> Gpu_type.Ne

(* [tid < N] / [tid >= N] agree inside a warp when [N] is a multiple of 32.
   [tid > N] / [tid <= N] agree when [N] is the last lane of a warp.
   Equality never agrees: one lane is not a warp. *)
let warp_aligned cmp n =
  match cmp with
  | Gpu_type.Lt | Gpu_type.Ge -> rem_warp n = 0
  | Gpu_type.Gt | Gpu_type.Le -> rem_warp n = warp - 1
  | Gpu_type.Eq | Gpu_type.Ne -> false

let analyze (kernel : kernel) =
  let values = Hashtbl.create 32 in
  let consts = Hashtbl.create 16 in
  let tids = Hashtbl.create 4 in
  let errors = ref [] in
  List.iter (fun a -> Hashtbl.add values a.value.id Grid) kernel.args;
  let get v = Option.value (Hashtbl.find_opt values v.id) ~default:Varying in
  let error i text =
    let loc = match operands i with v :: _ -> v.loc | [] -> Source_span.synthetic in
    errors := ("E_DIVERGENT_COLLECTIVE", Source_span.to_string loc ^ ": " ^ text) :: !errors
  in
  let tid_bound cmp a b =
    let side x y cmp =
      if Hashtbl.mem tids x.id then
        match Hashtbl.find_opt consts y.id with
        | Some n -> Some (cmp, n)
        | None -> None
      else None
    in
    match side a b cmp with
    | Some _ as hit -> hit
    | None -> side b a (flip_cmp cmp)
  in
  let record i =
    match i with
    | Const_i32 (v, n) -> Hashtbl.replace consts v.id n
    | Thread_idx_x v -> Hashtbl.replace tids v.id ()
    | _ -> ()
  in
  let fact_of i =
    match i with
    | Thread_idx_x _ | Global_idx_x _
    | Load_f32 _ | Load_f32_masked _ | Load_f32x4 _
    | Load_tensor _ | Load_tensor_masked _ | Remap_tensor _ | Shared_load_f32 _ ->
        Varying
    | Block_idx_x _ | Block_idx_y _ -> Cta
    | Warp_sum_f32 _ | Reduce_sum_f32 _ -> Subgroup
    | Compare (_, cmp, a, b) ->
        (match tid_bound cmp a b with
         | Some (cmp, n) when warp_aligned cmp n -> Subgroup
         | _ -> List.fold_left (fun f v -> join f (get v)) Grid (operands i))
    | _ -> List.fold_left (fun f v -> join f (get v)) Grid (operands i)
  in
  let note participation i =
    (match i with
     | Warp_sum_f32 _ | Reduce_sum_f32 _ ->
         if rank participation > rank Subgroup then
           error i "full subgroup reduction inside divergent control flow"
     | Barrier Execution.Cta ->
         if rank participation > rank Cta then
           error i "CTA barrier requires CTA-uniform participation"
     | Remap_tensor _ ->
         if rank participation > rank Cta then
           error i "tensor remap requires CTA-uniform participation"
     | _ -> ());
    record i;
    let fact = fact_of i in
    List.iter (fun v -> Hashtbl.replace values v.id (join participation fact)) (results i)
  in
  let rec walk participation = function
    | [] -> []
    | (Return _ :: _ as rest) -> rest
    | i :: rest ->
        let i =
          match i with
          | If (dst, c, a, b) | If_uni (dst, c, a, b) ->
              let active = join participation (get c) in
              let a = { a with body = walk active a.body } in
              let b = { b with body = walk active b.body } in
              Option.iter (fun d ->
                let yield r = Option.fold ~none:Grid ~some:get r.yield in
                Hashtbl.replace values d.id (join active (join (yield a) (yield b)))) dst;
              if (match i with If_uni _ -> true | _ -> false) || get c <=: Subgroup then
                If_uni (dst, c, a, b)
              else If (dst, c, a, b)
          | For (induction, limit, step, loop_body) ->
              let active = join participation (get limit) in
              (* The induction stays as uniform as its start, the trip limit,
                 and the region that reached the loop. *)
              Hashtbl.replace values induction.id
                (join participation (join (get induction) (get limit)));
              For (induction, limit, step, walk active loop_body)
          | _ ->
              note participation i;
              i
        in
        i :: walk participation rest
  in
  let body = walk Grid kernel.body in
  body, List.rev !errors

let check kernel = snd (analyze kernel)
let promote kernel = { kernel with body = fst (analyze kernel) }
