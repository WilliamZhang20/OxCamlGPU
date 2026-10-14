(* Scalar arithmetic and comparison. Native [int] arithmetic is folded and
   range-checked at compile time, since it must provably fit i32; the [i32_*]
   forms wrap like the hardware and are never range-checked. *)
open Typedtree
module K = Kernel_ast
module G = Gpu_type
open Typedtree_support
open Export_state

let float_node = function
  | "+." -> fun d a b -> K.Add_f32 (d, a, b)
  | "*." -> fun d a b -> K.Mul_f32 (d, a, b)
  | "-." -> fun d a b -> K.Binary_f32 (d, G.Subtract, a, b)
  | _ -> fun d a b -> K.Binary_f32 (d, G.Divide, a, b)

let float_binary : handler =
  on_positional (fun ctx -> function
    | [a; b] ->
        let av = num ctx a in
        let bv = num ctx b in
        produce ctx (fun d -> float_node ctx.op d av bv)
    | _ -> arity ctx)

let int_node = function
  | "+" | "i32_add" -> fun d a b -> K.Add_i32 (d, a, b)
  | "-" | "i32_sub" -> fun d a b -> K.Sub_i32 (d, a, b)
  | "i32_div" -> fun d a b -> K.Div_i32 (d, a, b)
  | "i32_rem" -> fun d a b -> K.Rem_i32 (d, a, b)
  | _ -> fun d a b -> K.Mul_i32 (d, a, b)

(* The result of folding two constant operands, or [None] when the operation
   has to be emitted. Division by zero and [min_int / -1] are never folded. *)
let fold_constants op a b =
  let div_overflow a b = a = i32_min && b = -1L in
  match op, a, b with
  | ("/" | "i32_div"), Some a, Some b when b <> 0L && not (div_overflow a b) ->
      Some (Int64.div a b)
  | "i32_rem", Some a, Some b when b <> 0L && not (div_overflow a b) ->
      Some (Int64.rem a b)
  | ("+" | "i32_add"), Some a, Some b -> Some (Int64.add a b)
  | ("-" | "i32_sub"), Some a, Some b -> Some (Int64.sub a b)
  | ("*" | "i32_mul"), Some a, Some b -> Some (Int64.mul a b)
  | _ -> None

(* Interval arithmetic for native [+], [-] and [*]. *)
let native_range loc op ar br =
  match ar, br with
  | Some (al, ah), Some (bl, bh) ->
      let lo, hi =
        match op with
        | "+" -> Int64.add al bl, Int64.add ah bh
        | "-" -> Int64.sub al bh, Int64.sub ah bl
        | _ ->
            let products =
              List.map (fun (x, y) -> Int64.mul x y)
                [al, bl; al, bh; ah, bl; ah, bh]
            in
            ( List.fold_left min Int64.max_int products
            , List.fold_left max Int64.min_int products )
      in
      checked loc (lo, hi)
  | _ ->
      fail loc
        "cannot prove native int arithmetic fits i32; use Int32 arithmetic"

let int_binary : handler =
  on_positional (fun ctx -> function
    | [a; b] ->
        let av, ar = arg ctx scalar a in
        let bv, br = arg ctx scalar b in
        let loc = ctx.exp.exp_loc in
        let wrapping = String.starts_with ~prefix:"i32_" ctx.op in
        (match fold_constants ctx.op (constant_range ar) (constant_range br) with
         | Some v ->
             let v, range =
               if wrapping then
                 let v = i32_wrap v in
                 v, Some (v, v)
               else v, checked loc (v, v)
             in
             produce ?range ctx (fun d -> K.Const_i32 (d, Int64.to_int v))
         | None when ctx.op = "/" ->
             fail loc
               "native int division is only supported for non-zero compile-time constants"
         | None ->
             let range =
               if wrapping then full_range
               else native_range loc ctx.op ar br
             in
             produce ?range ctx (fun d -> int_node ctx.op d av bv))
    | _ -> arity ctx)

let comparison = function
  | "=" -> G.Eq
  | "<>" -> G.Ne
  | "<" -> G.Lt
  | "<=" -> G.Le
  | ">" -> G.Gt
  | _ -> G.Ge

let holds cmp a b =
  match cmp with
  | G.Eq -> a = b
  | G.Ne -> a <> b
  | G.Lt -> a < b
  | G.Le -> a <= b
  | G.Gt -> a > b
  | G.Ge -> a >= b

let compare_values : handler =
  on_positional (fun ctx -> function
    | [a; b] ->
        let av, ar = arg ctx scalar a in
        let bv, br = arg ctx scalar b in
        let k = kind a.exp_env a.exp_loc a.exp_type in
        if not (is_scalar k) then
          fail_here ctx "only scalar comparisons are supported";
        let cmp = comparison ctx.op in
        if k = Boolean && cmp <> G.Eq && cmp <> G.Ne then
          fail_here ctx "only boolean equality comparisons are supported";
        (match constant_range ar, constant_range br with
         | Some a, Some b -> bool_const ctx (holds cmp a b)
         | _ -> produce ctx (fun d -> K.Compare (d, cmp, av, bv)))
    | _ -> arity ctx)

(* [Int32.of_int] and [Int32.to_int] are the identity on the i32 register. *)
let identity : handler =
  on_positional (fun ctx -> function
    | [v] -> eval ctx v
    | _ -> arity ctx)

let prims : (string * handler) list =
  List.map (fun op -> op, float_binary) ["+."; "*."; "-."; "/."]
  @ List.map (fun op -> op, int_binary)
      ["+"; "-"; "*"; "/"; "i32_add"; "i32_sub"; "i32_mul"; "i32_div"; "i32_rem"]
  @ List.map (fun op -> op, compare_values)
      ["="; "<>"; "<"; "<="; ">"; ">="]
  @ [ "i32_of_int", identity; "i32_to_int", identity ]
