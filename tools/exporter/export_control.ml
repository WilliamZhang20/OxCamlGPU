(* Control flow: conditionals (including the short-circuit operators) and
   constant or dynamic [for] loops. Constant tests fold, so only the taken arm
   is traced; that is also what lets a recursion terminate. *)
open Typedtree
module K = Kernel_ast
open Typedtree_support
open Export_state

(* Trace [yes] and [no] as separate regions and join them in a single [If]. A
   branch yields at most one scalar, whose range is the hull of the arms. *)
let conditional ctx condition yes no =
  let region f =
    let body, v = capture_region ctx.s f in
    let yield =
      match v with
      | Void -> None
      | Scalar (a, _) | Const_bool (a, _) -> Some a
      | Func _ -> fail_here ctx "functions cannot be branch results"
      | Memory _ | Shared _ | Mbarrier _ | Tensor_map _ | Wgmma_acc _ ->
          fail_here ctx "non-scalar branch results are unsupported"
    in
    { K.body; yield }, v
  in
  let a, av = region yes in
  let b, bv = region no in
  (* Only the two scalar arms reach the join; spelling the rest out means the
     compiler rechecks that when [binding] grows a constructor. *)
  let arm_range = function
    | Scalar (_, r) -> r
    | Const_bool _ -> None
    | Memory _ | Shared _ | Mbarrier _ | Tensor_map _ | Wgmma_acc _ | Func _
    | Void -> fail_here ctx "branch result is not a scalar"
  in
  let dst, result =
    match av, bv with
    | Void, Void -> None, Void
    | (Scalar _ | Const_bool _), (Scalar _ | Const_bool _) ->
        let d = fresh ctx.s in
        let range =
          match arm_range av, arm_range bv with
          | Some (al, ah), Some (bl, bh) -> Some (min al bl, max ah bh)
          | _ -> None
        in
        ( Some (d, gpu_ty (kind ctx.exp.exp_env ctx.exp.exp_loc ctx.exp.exp_type))
        , Scalar (K.Value d, range) )
    | _ -> fail_here ctx "conditional arms have incompatible results"
  in
  emit ctx (K.If (dst, condition, a, b));
  result

(* [if c then x else y] where [c] may fold. *)
let if_then_else ctx cond yes no =
  let else_branch () = Option.fold ~none:Void ~some:(eval ctx) no in
  match eval ctx cond with
  | Const_bool (_, true) -> eval ctx yes
  | Const_bool (_, false) -> else_branch ()
  | c ->
      let c, _ = scalar cond.exp_loc c in
      conditional ctx c (fun () -> eval ctx yes) else_branch

(* [&&], [||] and [not] short-circuit through [conditional], so the operand
   that is not taken is never evaluated on the device. *)
let boolean_literal ctx v () = produce ctx (fun d -> K.Const_bool (d, v))

let short_circuit : handler =
  on_positional (fun ctx -> function
    | [a; b] ->
        let a = num ctx a in
        let rhs () = eval ctx b in
        if ctx.op = "&&" then conditional ctx a rhs (boolean_literal ctx false)
        else conditional ctx a (boolean_literal ctx true) rhs
    | _ -> arity ctx)

let negation : handler =
  on_positional (fun ctx -> function
    | [a] ->
        let a = num ctx a in
        conditional ctx a (boolean_literal ctx false) (boolean_literal ctx true)
    | _ -> arity ctx)

let prims : (string * handler) list =
  [ "&&", short_circuit; "||", short_circuit; "not", negation ]

(* OCaml's [for i = lo to hi] is inclusive; [K.For] is exclusive with a step.
   Constant bounds up to [unroll_bounds] iterations are copied, so an index
   such as [4 * i] stays a compile-time register select. Constants that
   folding leaves with no users are dropped at the end of lowering. *)
let for_loop ctx ~index ~from ~until body =
  let lo, lo_range = arg ctx scalar from in
  let hi, hi_range = arg ctx scalar until in
  let run_body env = unit body.exp_loc (ctx.recur env body) in
  let dynamic () =
    let limit =
      match constant_range hi_range with
      | Some h when h < i32_max ->
          value ctx (fun d -> K.Const_i32 (d, Int64.to_int h + 1))
      | Some _ ->
          fail_here ctx
            "for upper bound is Int32.max_int; an exclusive limit would overflow"
      | None ->
          let one = value ctx (fun d -> K.Const_i32 (d, 1)) in
          value ctx (fun d -> K.Add_i32 (d, hi, one))
    in
    let ind = fresh ctx.s in
    let loop_body, () =
      capture_region ctx.s (fun () ->
        run_body ((index, Scalar (K.Value ind, full_range)) :: ctx.env))
    in
    emit ctx (K.For (ind, lo, limit, 1, { K.body = loop_body; yield = None }));
    Void
  in
  match constant_range lo_range, constant_range hi_range with
  | Some first, Some last ->
      let trips =
        if last < first then 0L else Int64.succ (Int64.sub last first)
      in
      if trips > Int64.of_int unroll_bounds then dynamic ()
      else begin
        for i = Int64.to_int first to Int64.to_int last do
          let i64 = Int64.of_int i in
          let k = value ctx (fun d -> K.Const_i32 (d, i)) in
          run_body ((index, Scalar (k, Some (i64, i64))) :: ctx.env)
        done;
        Void
      end
  | _ -> dynamic ()
