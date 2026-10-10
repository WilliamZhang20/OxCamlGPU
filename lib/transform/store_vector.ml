(* Fuse consecutive scalar f32 stores into one vector store.

   Two facts have to hold before a run of [Store_f32] can become a
   [Store_f32x2] or [Store_f32x4]:

   - the element indices are consecutive, which needs the *difference* between
     two index expressions, and
   - the first index is a multiple of the vector width, because PTX vector
     stores require natural alignment and a misaligned one faults.

   So each i32 value carries two abstractions. A canonical affine form over
   opaque atoms settles differences; a congruence [v = residue (mod modulus)]
   settles alignment. Atoms are keyed by structure rather than by value id, so
   two separately computed but identical subexpressions (a row stride
   recomputed per store, say) unify without a prior CSE pass.

   Nothing here changes what a kernel computes: it changes how many
   instructions carry the same bytes to the same addresses. *)

open Ir

let rec gcd a b = if b = 0 then abs a else gcd b (a mod b)

(* --- congruence: v = residue (mod modulus). modulus 0 means exactly residue,
   so it divides everything; modulus 1 means nothing is known. --- *)

type congruence = { modulus : int; residue : int }

let unknown = { modulus = 1; residue = 0 }
let exactly n = { modulus = 0; residue = n }

let normalize c =
  if c.modulus = 0 then c
  else
    let m = abs c.modulus in
    if m <= 1 then unknown
    else { modulus = m; residue = ((c.residue mod m) + m) mod m }

let add_cong a b =
  normalize { modulus = gcd a.modulus b.modulus; residue = a.residue + b.residue }

let sub_cong a b =
  normalize { modulus = gcd a.modulus b.modulus; residue = a.residue - b.residue }

(* a = ma*s + ra and b = mb*t + rb, so a*b = ma*mb*st + ma*s*rb + mb*t*ra
   + ra*rb. Every term but the last is divisible by the gcd below. *)
let mul_cong a b =
  if a.modulus = 0 && b.modulus = 0 then exactly (a.residue * b.residue)
  else
    let m =
      gcd (gcd (a.modulus * b.modulus) (a.modulus * b.residue))
        (b.modulus * a.residue)
    in
    normalize { modulus = m; residue = a.residue * b.residue }

(* [v] is a multiple of [width]. *)
let divides width c =
  width > 0 && c.residue mod width = 0
  && (c.modulus = 0 || c.modulus mod width = 0)

(* --- affine form: constant + sum of coefficient * atom, atoms sorted --- *)

type affine = { constant : int; terms : (string * int) list }

let constant_form n = { constant = n; terms = [] }

let merge_terms a b =
  let table = Hashtbl.create 8 in
  List.iter (fun (k, c) -> Hashtbl.replace table k c) a;
  List.iter
    (fun (k, c) ->
      let current = try Hashtbl.find table k with Not_found -> 0 in
      Hashtbl.replace table k (current + c))
    b;
  Hashtbl.fold (fun k c acc -> if c = 0 then acc else (k, c) :: acc) table []
  |> List.sort (fun (a, _) (b, _) -> compare a b)

let add_form a b =
  { constant = a.constant + b.constant; terms = merge_terms a.terms b.terms }

let negate_form a =
  { constant = -a.constant; terms = List.map (fun (k, c) -> (k, -c)) a.terms }

let sub_form a b = add_form a (negate_form b)

let scale_form n a =
  if n = 0 then constant_form 0
  else
    { constant = a.constant * n;
      terms = List.map (fun (k, c) -> (k, c * n)) a.terms }

let string_of_form f =
  String.concat "+"
    (string_of_int f.constant
     :: List.map (fun (k, c) -> string_of_int c ^ "*" ^ k) f.terms)

(* [difference b a] is [Some n] when b - a is the constant n. *)
let difference b a =
  let d = sub_form b a in
  if d.terms = [] then Some d.constant else None

(* --- abstraction of a kernel's i32 values --- *)

type analysis = {
  defs : (int, instr) Hashtbl.t;
  cong_memo : (int, congruence) Hashtbl.t;
  form_memo : (int, affine) Hashtbl.t;
  atom_memo : (int, string) Hashtbl.t;
}

let sub_bodies = function
  | If (_, _, yes, no) | If_uni (_, _, yes, no) -> [ yes.body; no.body ]
  | For (_, _, _, body) -> [ body ]
  | _ -> []

let analyze kernel =
  let defs = Hashtbl.create 256 in
  let rec walk body =
    List.iter
      (fun instruction ->
        List.iter
          (fun value -> Hashtbl.replace defs value.id instruction)
          (results instruction);
        List.iter walk (sub_bodies instruction))
      body
  in
  walk kernel.body;
  { defs;
    cong_memo = Hashtbl.create 256;
    form_memo = Hashtbl.create 256;
    atom_memo = Hashtbl.create 256 }

let rec congruence a value =
  match Hashtbl.find_opt a.cong_memo value.id with
  | Some c -> c
  | None ->
      (* SSA is acyclic, but seed the memo anyway so a malformed body cannot
         spin here. *)
      Hashtbl.replace a.cong_memo value.id unknown;
      let c =
        match Hashtbl.find_opt a.defs value.id with
        | Some (Const_i32 (_, n)) -> exactly n
        | Some (Add_i32 (_, x, y)) -> add_cong (congruence a x) (congruence a y)
        | Some (Sub_i32 (_, x, y)) -> sub_cong (congruence a x) (congruence a y)
        | Some (Mul_i32 (_, x, y)) -> mul_cong (congruence a x) (congruence a y)
        | _ -> unknown
      in
      Hashtbl.replace a.cong_memo value.id c;
      c

and form a value =
  match Hashtbl.find_opt a.form_memo value.id with
  | Some f -> f
  | None ->
      Hashtbl.replace a.form_memo value.id
        { constant = 0; terms = [ ("v" ^ string_of_int value.id, 1) ] };
      let f =
        match Hashtbl.find_opt a.defs value.id with
        | Some (Const_i32 (_, n)) -> constant_form n
        | Some (Add_i32 (_, x, y)) -> add_form (form a x) (form a y)
        | Some (Sub_i32 (_, x, y)) -> sub_form (form a x) (form a y)
        | Some (Mul_i32 (_, x, y)) ->
            let fx = form a x and fy = form a y in
            if fx.terms = [] then scale_form fx.constant fy
            else if fy.terms = [] then scale_form fy.constant fx
            else { constant = 0; terms = [ (atom a value, 1) ] }
        | _ -> { constant = 0; terms = [ (atom a value, 1) ] }
      in
      Hashtbl.replace a.form_memo value.id f;
      f

(* A structural key, so equal expressions computed twice share one atom. *)
and atom a value =
  match Hashtbl.find_opt a.atom_memo value.id with
  | Some k -> k
  | None ->
      Hashtbl.replace a.atom_memo value.id ("v" ^ string_of_int value.id);
      let binary name x y =
        name ^ "(" ^ string_of_form (form a x) ^ "," ^ string_of_form (form a y)
        ^ ")"
      in
      let k =
        match Hashtbl.find_opt a.defs value.id with
        | Some (Mul_i32 (_, x, y)) -> binary "mul" x y
        | Some (Div_i32 (_, x, y)) -> binary "div" x y
        | Some (Rem_i32 (_, x, y)) -> binary "rem" x y
        | Some (Thread_idx_x _) -> "tid.x"
        | Some (Global_idx_x _) -> "gid.x"
        | Some (Block_idx_x _) -> "ctaid.x"
        | Some (Block_idx_y _) -> "ctaid.y"
        | _ -> "v" ^ string_of_int value.id
      in
      Hashtbl.replace a.atom_memo value.id k;
      k

(* --- fusion --- *)

(* Instructions a store may be moved across: no memory effect, no control
   flow. Everything else blocks, conservatively. *)
let movable_across = function
  | Const_i32 _ | Const_f32 _ | Const_bool _ | Thread_idx_x _ | Global_idx_x _
  | Block_idx_x _ | Block_idx_y _ | Gep_f32 _ | Add_i32 _ | Sub_i32 _
  | Mul_i32 _ | Div_i32 _ | Rem_i32 _ | Add_f32 _ | Mul_f32 _ | Mad_f32 _
  | Compare _ ->
      true
  | _ -> false

type candidate = {
  position : int;
  base_id : int;
  pointer : value;
  index : value;
  stored : value;
}

let vector_store pointer values =
  match values with
  | [ a; b ] -> Some (Store_f32x2 (pointer, a, b))
  | [ a; b; c; d ] -> Some (Store_f32x4 (pointer, a, b, c, d))
  | _ -> None

(* Rewrite one straight-line instruction list. *)
let fuse_straight_line a body =
  let instructions = Array.of_list body in
  let candidates =
    Array.to_list instructions
    |> List.mapi (fun position instruction -> (position, instruction))
    |> List.filter_map (fun (position, instruction) ->
           match instruction with
           | Store_f32 (pointer, stored) -> (
               match Hashtbl.find_opt a.defs pointer.id with
               | Some (Gep_f32 (_, base, index)) ->
                   Some { position; base_id = base.id; pointer; index; stored }
               | _ -> None)
           | _ -> None)
  in
  (* Fusing moves a run's earlier stores forward to the last one's position,
     so nothing in between may touch memory. The run's own stores are the
     exception, since they are what is being merged. No *other* store can land
     inside a run: runs are grown from candidates adjacent in program order,
     so an intervening store would already have ended the run by failing the
     base or index test below. *)
  let clear_between run_positions first last =
    let rec check i =
      i >= last
      || ((movable_across instructions.(i) || List.mem i run_positions)
          && check (i + 1))
    in
    check (first + 1)
  in
  (* Maximal runs of candidates that are consecutive in both program order and
     element index. *)
  let rec runs = function
    | [] -> []
    | first :: rest ->
        let rec extend acc last = function
          | next :: tail
            when next.base_id = last.base_id
                 && difference (form a next.index) (form a last.index) = Some 1
                 && clear_between
                      (List.map (fun c -> c.position) (next :: acc))
                      first.position next.position ->
              extend (next :: acc) next tail
          | remaining -> (List.rev acc, remaining)
        in
        let run, remaining = extend [ first ] first rest in
        run :: runs remaining
  in
  let edits = Hashtbl.create 16 in
  let rec chunk = function
    | [] -> ()
    | run ->
        let length = List.length run in
        let start = List.hd run in
        let width =
          if length >= 4 && divides 4 (congruence a start.index) then 4
          else if length >= 2 && divides 2 (congruence a start.index) then 2
          else 1
        in
        if width = 1 then chunk (List.tl run)
        else begin
          let taken = List.filteri (fun i _ -> i < width) run in
          let rest = List.filteri (fun i _ -> i >= width) run in
          (match vector_store start.pointer (List.map (fun c -> c.stored) taken) with
           | Some fused ->
               (* The fused store lands where the last of the group was, so
                  every stored value is already defined. *)
               let last = List.nth taken (width - 1) in
               List.iter
                 (fun c ->
                   if c.position <> last.position then
                     Hashtbl.replace edits c.position None)
                 taken;
               Hashtbl.replace edits last.position (Some fused)
           | None -> ());
          chunk rest
        end
  in
  List.iter chunk (runs candidates);
  if Hashtbl.length edits = 0 then body
  else
    Array.to_list instructions
    |> List.mapi (fun position instruction ->
           match Hashtbl.find_opt edits position with
           | Some replacement -> replacement
           | None -> Some instruction)
    |> List.filter_map (fun x -> x)

let rec fuse_body a body =
  let body =
    List.map
      (fun instruction ->
        match instruction with
        | If (dst, condition, yes, no) ->
            If (dst, condition, fuse_region a yes, fuse_region a no)
        | If_uni (dst, condition, yes, no) ->
            If_uni (dst, condition, fuse_region a yes, fuse_region a no)
        | For (induction, limit, step, nested) ->
            For (induction, limit, step, fuse_body a nested)
        | other -> other)
      body
  in
  fuse_straight_line a body

and fuse_region a region = { region with body = fuse_body a region.body }

(* Address arithmetic that fusion leaves unused is pure and unreferenced, so
   ptxas drops it; doing it here would mean tracking region yields, which are
   uses that [Ir.operands] does not report. *)

let fuse kernel = { kernel with body = fuse_body (analyze kernel) kernel.body }
