(* Lower one named top-level kernel of a Typedtree structure to [Kernel_ast]. *)
open Typedtree
module K = Kernel_ast
module G = Gpu_type
open Typedtree_support
open Export_state

let threads_attribute binding =
  let open Parsetree in
  let payload_int loc = function
    | PStr
        [{ pstr_desc =
             Pstr_eval
               ( { pexp_desc =
                     Pexp_constant
                       { pconst_desc = Pconst_integer (text, None); _ }
                 ; _ }
               , _ )
         ; _ }] ->
        (match int_of_string_opt text with
         | Some n -> n
         | None -> fail loc "@@gpu.threads integer is out of range")
    | _ ->
        fail loc "@@gpu.threads expects an integer, e.g. [@@gpu.threads 288]"
  in
  match
    List.filter_map
      (fun attr ->
        if attr.attr_name.txt = "gpu.threads" then
          Some (payload_int attr.attr_loc attr.attr_payload)
        else None)
      binding.vb_attributes
  with
  | [] -> None
  | [n] when n >= 1 && n <= 1024 -> Some n
  | [_] -> fail binding.vb_loc "@@gpu.threads must be in 1..1024"
  | _ -> fail binding.vb_loc "duplicate @@gpu.threads"

let defines target binding =
  match binding.vb_pat.pat_desc with
  | Tpat_var { name; _ } -> name.txt = target
  | _ -> false

let find_binding target structure =
  let found =
    List.find_map
      (fun item ->
        match item.str_desc with
        | Tstr_value (_, bindings) -> List.find_opt (defines target) bindings
        | _ -> None)
      structure.str_items
  in
  match found with
  | Some b -> b
  | None -> fail Location.none "kernel binding not found"

(* Structure-level functions defined before [target]. A specialized kernel is
   an application of one of these ([hopper_gemm ~bm:128 ...]); evaluating the
   application inlines the constants, and the returned function is the entry. *)
let prior_functions target structure =
  let value_bindings =
    List.concat_map
      (fun item ->
        match item.str_desc with
        | Tstr_value (_, bindings) -> bindings
        | _ -> [])
      structure.str_items
  in
  let rec take_before = function
    | b :: _ when defines target b -> []
    | b :: rest -> b :: take_before rest
    | [] -> []
  in
  List.fold_left
    (fun env binding ->
      match binding.vb_pat.pat_desc, binding.vb_expr.exp_desc with
      | Tpat_var { id; _ }, Texp_function _ ->
          (match Export_expr.expression (create_state ()) env binding.vb_expr with
           | Func _ as fn -> (id, fn) :: env
           | _ -> env)
      | _ -> env)
    [] (take_before value_bindings)

(* Kernel formals of a direct kernel are read off the parameter patterns. *)
let direct_formal loc i param =
  let k =
    match param.fp_kind with
    | Tparam_pat p -> kind p.pat_env p.pat_loc p.pat_type
    | _ -> fail loc "optional parameters unsupported"
  in
  let value =
    match k with
    | Buffer _ -> Memory (K.Buffer_arg i)
    | Tensor_map _ -> Tensor_map (K.Arg i)
    | Unit -> fail loc "unit kernel argument unsupported"
    | Shared _ | Mbarrier ->
        fail loc "shared/mbarrier cannot be kernel formals"
    | Native_int | Int32 -> Scalar (K.Arg i, full_range)
    | Int64 | Gmma_desc _ | Float | Boolean -> Scalar (K.Arg i, None)
    | Wgmma_acc -> fail loc "wgmma_acc cannot be a kernel formal"
  in
  param.fp_param, value

(* Kernel formals of a specialized kernel are read off the entry signature. *)
let signature_formal loc i id (arg : K.argument) =
  let value =
    match arg.slot.ty with
    | G.MemRef (_, _, G.Global) | G.Ptr (_, G.Global) -> Memory (K.Buffer_arg i)
    | G.Tensor_map -> Tensor_map (K.Arg i)
    | G.I32 -> Scalar (K.Arg i, full_range)
    | G.F32 | G.Bool | G.U64 -> Scalar (K.Arg i, None)
    | _ -> fail loc "unsupported kernel argument"
  in
  id, value

let entry_signature binding =
  let tenv = binding.vb_expr.exp_env and loc = binding.vb_expr.exp_loc in
  let rec signature i typ last =
    match Types.get_desc typ with
    | Types.Tpoly (t, _) -> signature i t last
    | Types.Tarrow ((_, am, rm), a, b, _) ->
        let a = { K.index = i; slot = slot tenv loc a am } in
        let rest, result = signature (i + 1) b (Some rm) in
        a :: rest, result
    | _ ->
        match last with
        | Some m -> [], slot tenv loc typ m
        | None -> fail loc "kernel must be a function"
  in
  let args, result = signature 0 binding.vb_pat.pat_type None in
  if result.ty <> G.Unit then fail loc "GPU entry kernels must return unit";
  args, result

let compile target structure =
  let binding = find_binding target structure in
  let loc = binding.vb_expr.exp_loc in
  let args, result = entry_signature binding in
  let s = create_state () in
  let trace env body =
    unit body.exp_loc (Export_expr.expression s env body)
  in
  (match binding.vb_expr.exp_desc with
   | Texp_function { params; body = Tfunction_body body; _ } ->
       (* Direct kernels do not see earlier structure bindings. A user
          operator with the same name as a primitive must stay rejected. *)
       trace (List.mapi (direct_formal loc) params) body
   | _ ->
       let env = prior_functions target structure in
       let fn =
         match Export_expr.expression s env binding.vb_expr with
         | Func fn -> fn
         | _ ->
             fail loc
               "kernel binding must be a function or a fully applied specialization of one"
       in
       if List.length fn.params <> List.length args then
         fail loc "kernel function arity does not match its type";
       let formals =
         List.mapi
           (fun i ((_, id), arg) -> signature_formal loc i id arg)
           (List.combine fn.params args)
       in
       trace (formals @ fn.env) fn.body);
  { K.name = target
  ; args
  ; result
  ; body = Export_dce.eliminate_unused_constants (List.rev s.body)
  ; threads = threads_attribute binding
  }
