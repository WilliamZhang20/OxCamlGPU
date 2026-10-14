(* Float intrinsics: conversions, unary and binary functions, fused
   multiply-add, and the warp reduction. *)
open Typedtree
module K = Kernel_ast
module G = Gpu_type
open Typedtree_support
open Export_state

let float_of_int : handler =
  on_positional (fun ctx -> function
    | [a] ->
        let a = num ctx a in
        produce ctx (fun d -> K.F32_of_i32 (d, a))
    | _ -> arity ctx)

let unary_function = function
  | "rsqrt_approx" -> G.Rsqrt_approx
  | "exp2_approx" -> G.Exp2_approx
  | "log2_approx" -> G.Log2_approx
  | "tanh_approx" -> G.Tanh_approx
  | "recip_approx" -> G.Recip_approx
  | "sqrt" -> G.Sqrt
  | "abs_float" -> G.Absolute
  | _ -> G.Negate

let unary : handler =
  on_positional (fun ctx -> function
    | [a] ->
        let fn = unary_function ctx.op in
        let a = num ctx a in
        produce ctx (fun d -> K.Unary_f32 (d, fn, a))
    | _ -> arity ctx)

let extremum : handler =
  on_positional (fun ctx -> function
    | [a; b] ->
        let fn = if ctx.op = "float_max" then G.Maximum else G.Minimum in
        let a = num ctx a in
        let b = num ctx b in
        produce ctx (fun d -> K.Binary_f32 (d, fn, a, b))
    | _ -> arity ctx)

let mad_f32 : handler =
  on_positional (fun ctx -> function
    | [acc; a; b] ->
        let acc = num ctx acc in
        let a = num ctx a in
        let b = num ctx b in
        produce ctx (fun d -> K.Mad_f32 (d, acc, a, b))
    | _ -> arity ctx)

let warp_sum_f32 : handler =
  on_positional (fun ctx -> function
    | [v] ->
        let v = num ctx v in
        produce ctx (fun d -> K.Warp_reduce_sum_f32 (d, v))
    | _ -> arity ctx)

let prims : (string * handler) list =
  List.map (fun op -> op, unary)
    [ "rsqrt_approx"; "exp2_approx"; "log2_approx"; "tanh_approx"
    ; "recip_approx"; "sqrt"; "abs_float"; "~-." ]
  @ [ "float_of_int", float_of_int
    ; "float_max", extremum
    ; "float_min", extremum
    ; "mad_f32", mad_f32
    ; "warp_sum_f32", warp_sum_f32 ]
