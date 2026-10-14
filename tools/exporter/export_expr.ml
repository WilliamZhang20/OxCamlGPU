(* Walk a Typedtree expression, evaluating it at compile time and emitting the
   device instructions it denotes. Names bind to [binding]s; calls to local
   functions are inlined; calls to GPU primitives dispatch through
   [Export_prims]. *)
open Typedtree
module K = Kernel_ast
open Typedtree_support
open Export_state

let function_params loc params =
  List.map
    (fun (p : function_param) ->
      (match p.fp_arg_label with
       | Optional _ | Position _ -> fail loc "optional parameters unsupported"
       | Nolabel | Labelled _ -> ());
      (match p.fp_kind with
       | Tparam_pat { pat_desc = Tpat_var _ | Tpat_any; _ } -> ()
       | Tparam_pat { pat_desc = Tpat_construct (_, c, _, [], _); _ }
         when c.cstr_name = "()" -> ()
       | Tparam_pat pat ->
           fail pat.pat_loc "only simple function parameters are supported"
       | _ -> fail loc "optional parameters unsupported");
      p.fp_arg_label, p.fp_param)
    params

(* Bind the evaluated call arguments to the callee's parameters. Positional
   arguments share the label [Nolabel], so those match in order; labelled
   arguments match by name. *)
let bind_parameters loc (fn : func) args =
  let missing () = fail loc "missing function argument" in
  let not_fully_applied () = fail loc "functions must be fully applied" in
  if List.length args <> List.length fn.params then not_fully_applied ();
  let step (pending, env) (label, pid) =
    match (label : Types.arg_label) with
    | Nolabel ->
        (match pending with
         | (_, v) :: rest -> rest, (pid, v) :: env
         | [] -> missing ())
    | Labelled name ->
        (match List.find_opt (fun (got, _) -> got = Types.Labelled name) args with
         | Some (_, v) -> pending, (pid, v) :: env
         | None -> missing ())
    | Optional _ | Position _ -> fail loc "optional arguments unsupported"
  in
  let positionals = List.filter (fun (l, _) -> l = Types.Nolabel) args in
  let leftover, env = List.fold_left step (positionals, fn.env) fn.params in
  if leftover <> [] then not_fully_applied ();
  env

let rec expression s env exp =
  if s.nesting >= max_expression_nesting then
    fail exp.exp_loc
      (Printf.sprintf
         "expression nested deeper than %d (OXGPU_MAX_NESTING); split the \
          kernel into smaller definitions"
         max_expression_nesting);
  s.nesting <- s.nesting + 1;
  let traced = expression_node s env exp in
  s.nesting <- s.nesting - 1;
  traced

and context s env exp op = { s; env; exp; op; recur = expression s }

and expression_node s env exp =
  let ctx = context s env exp "" in
  let constant ?range node = produce ?range ctx node in
  match exp.exp_desc with
  | Texp_constant (Const_int n) ->
      let v = Int64.of_int n in
      ignore (checked exp.exp_loc (v, v));
      constant ~range:(v, v) (fun d -> K.Const_i32 (d, n))
  | Texp_constant (Const_int32 n) ->
      let v = Int64.of_int32 n in
      constant ~range:(v, v) (fun d -> K.Const_i32 (d, Int32.to_int n))
  | Texp_constant (Const_int64 n) ->
      (* Used for gmma descriptor results typed as int64 / u64. *)
      if n < 0L then fail exp.exp_loc "negative int64 literal unsupported";
      constant (fun d -> K.Const_i32 (d, Int64.to_int n))
  | Texp_constant (Const_float n) ->
      constant (fun d -> K.Const_f32 (d, float_of_string n))
  | Texp_ident { path = Path.Pident id; _ } ->
      (match lookup env id with
       | Some v -> v
       | None -> fail exp.exp_loc "unbound GPU value")
  | Texp_ident _ -> fail exp.exp_loc "unsupported nonlocal value"
  | Texp_open (_, e) -> expression s env e
  | Texp_construct (_, c, _, [], _) ->
      (match kind exp.exp_env exp.exp_loc exp.exp_type, c.cstr_name with
       | Unit, "()" -> Void
       | Boolean, "true" -> bool_const ctx true
       | Boolean, "false" -> bool_const ctx false
       | _ -> fail exp.exp_loc "unsupported constructor")
  | Texp_let (Recursive, bindings, body) ->
      expression s (recursive_bindings env bindings) body
  | Texp_let (Nonrecursive, bindings, body) ->
      let_bindings ctx bindings body
  | Texp_sequence (a, _, b) ->
      unit_arg ctx a;
      expression s env b
  | Texp_ifthenelse (c, a, b) -> Export_control.if_then_else ctx c a b
  | Texp_function { params; body = Tfunction_body fbody; _ } ->
      Func { params = function_params exp.exp_loc params; body = fbody; env }
  | Texp_function _ ->
      fail exp.exp_loc "pattern-matching functions are unsupported"
  | Texp_apply (callee, xs, _, _, _, _) -> apply ctx callee xs
  | Texp_for { for_id; for_from; for_to; for_dir = Upto; for_body; _ } ->
      Export_control.for_loop ctx ~index:for_id ~from:for_from ~until:for_to
        for_body
  | Texp_for _ -> fail exp.exp_loc "only ascending for-loops are supported"
  | _ -> fail exp.exp_loc "unsupported GPU expression"

(* A recursive function is fine as long as its base case folds, which it does
   when the controlling test is a compile-time constant: the conditional
   traces only the taken arm, so the inlining terminates. [inline_call]
   rebinds the callee's own name inside its body, which is what makes the
   recursive call resolve. *)
and recursive_bindings env bindings =
  List.fold_left
    (fun env binding ->
      match binding.vb_pat.pat_desc, binding.vb_expr.exp_desc with
      | Tpat_var { id; _ },
        Texp_function { params; body = Tfunction_body fbody; _ } ->
          let params = function_params binding.vb_expr.exp_loc params in
          (id, Func { params; body = fbody; env }) :: env
      | Tpat_var _, _ ->
          fail binding.vb_expr.exp_loc "a recursive binding must be a function"
      | _ -> fail binding.vb_pat.pat_loc "unsupported recursive pattern")
    env bindings

and let_bindings ctx bindings body =
  let rec bind env = function
    | [] -> expression ctx.s env body
    | binding :: rest ->
        let at_binding = context ctx.s env binding.vb_expr "" in
        (match binding.vb_pat.pat_desc, binding.vb_expr.exp_desc with
         | Tpat_tuple pats, Texp_apply (callee, xs, _, _, _, _) ->
             bind (vector_tuple at_binding binding callee xs pats @ env) rest
         | _ ->
             let v = expression ctx.s env binding.vb_expr in
             (match binding.vb_pat.pat_desc with
              | Tpat_var { id; _ } -> bind ((id, v) :: env) rest
              | Tpat_any -> bind env rest
              | _ ->
                  fail binding.vb_pat.pat_loc
                    "only simple kernel let bindings are supported"))
  in
  bind ctx.env bindings

(* [let (a, b, c, d) = load_f32x4 ...]: the one tuple binding kernels use. *)
and vector_tuple ctx binding callee xs pats =
  let op = primitive callee in
  let names =
    List.map
      (function
        | _, { pat_desc = Tpat_var { id; _ }; _ } -> id
        | _, p -> fail p.pat_loc "f32x4 must bind four names")
      pats
  in
  match op, names with
  | ("load_f32x4" | "shared_load_f32x4"), [a; b; c; d] ->
      let ia, ib, ic, id = Export_prims_memory.load_vector4 { ctx with op } xs in
      let register r = Scalar (K.Value r, None) in
      [ a, register ia; b, register ib; c, register ic; d, register id ]
  | _ -> fail binding.vb_pat.pat_loc "only f32x4 tuple bindings are supported"

and apply ctx callee xs =
  match inline_call ctx callee xs with
  | Some v -> v
  | None ->
      let op = primitive callee in
      (match Export_prims.find op with
       | Some handler -> handler { ctx with op } xs
       | None -> fail ctx.exp.exp_loc ("unsupported GPU call/arity: " ^ op))

(* A call to a function bound in the kernel is inlined at the call site. *)
and inline_call ctx callee xs =
  let s = ctx.s and loc = ctx.exp.exp_loc in
  match callee.exp_desc with
  | Texp_ident { path = Path.Pident id; _ } ->
      (match lookup ctx.env id with
       | Some (Func fn) ->
           let args =
             List.map
               (function
                 | (Optional _ | Position _), _ ->
                     fail loc "optional arguments unsupported"
                 | _, Omitted _ -> fail loc "functions must be fully applied"
                 | label, Arg (a, _) -> label, eval ctx a)
               xs
           in
           (* Rebinding the callee under its own name is what lets a function
              call itself; a non-recursive one simply never looks it up. *)
           let body_env = (id, Func fn) :: bind_parameters loc fn args in
           if s.depth >= max_inline_depth then
             fail loc
               (Printf.sprintf
                  "call nested deeper than %d (OXGPU_MAX_INLINE_DEPTH); a \
                   recursion whose base case is not a compile-time constant \
                   cannot be inlined"
                  max_inline_depth);
           s.depth <- s.depth + 1;
           let v = expression s body_env fn.body in
           s.depth <- s.depth - 1;
           Some v
       | _ -> None)
  | _ -> None
