(* Version-matched OxCaml compiler-libs adapter. The output is a deliberately
   small, stable interchange format consumed by Oxcaml_frontend. *)
open Typedtree

let fail message = prerr_endline message; exit 2

let tokens mode =
  String.split_on_char ',' mode |> List.map String.trim

let choose options values fallback =
  match List.find_opt (fun option -> List.mem option values) options with
  | Some value -> value
  | None -> fallback

let serialize_arrow_mode mode =
  let modes = Mode.With_locality.to_const_exn mode in
  let values = tokens (Format_doc.asprintf "%a" Mode.With_locality.Const.print modes) in
  [ choose ["unique"; "aliased"] values "aliased";
    choose ["local"; "global"] values "global";
    choose ["portable"; "nonportable"] values "nonportable";
    choose ["read_write"; "read"; "write"; "immutable"] values "read_write" ]
  |> String.concat "\t"

(* Keep the exporter independent of OxCamlGPU's runtime modules. *)
type kernel_expr =
  | Arg of int
  | Thread_idx_x
  | Load_f32 of int * kernel_expr
  | Add_f32 of kernel_expr * kernel_expr
  | Mul_f32 of kernel_expr * kernel_expr
type binding = Buffer of int | Value of kernel_expr

let rec emit_expr = function
  | Arg i -> "arg " ^ string_of_int i
  | Thread_idx_x -> "thread_idx_x"
  | Load_f32 (i, ix) -> "load " ^ string_of_int i ^ " " ^ emit_expr ix
  | Add_f32 (a,b) -> "add " ^ emit_expr a ^ " " ^ emit_expr b
  | Mul_f32 (a,b) -> "mul " ^ emit_expr a ^ " " ^ emit_expr b

let path_last path =
  let full_name = Path.name path in
  if String.ends_with ~suffix:"+." full_name then "+."
  else if String.ends_with ~suffix:"*." full_name then "*."
  else match List.rev (String.split_on_char '.' full_name) with
    | name :: _ -> name | [] -> ""

let rec identifier env path = match path with
  | Path.Pident id ->
      (match List.find_opt (fun (bound,_) -> Ident.same bound id) env with
       | Some (_, value) -> value
       | None -> fail ("unbound kernel value " ^ Ident.name id))
  | _ -> fail ("unsupported qualified value " ^ Path.name path)

let args_of_apply args = List.map (function
  | (Nolabel, Typedtree.Arg (arg, _)) -> arg
  | _ -> fail "only positional, fully applied GPU operations are supported") args

let rec expression env exp = match exp.exp_desc with
  | Texp_ident { path; _ } -> (match identifier env path with Value expr -> expr | Buffer _ -> fail "buffer used as scalar")
  | Texp_open (_, body) -> expression env body
  | Texp_apply (callee, args, _, _, _, _) ->
      let name = match callee.exp_desc with
        | Texp_ident { path; _ } -> path_last path
        | _ -> fail "GPU call target must be a resolved identifier" in
      let args = args_of_apply args in
      (match name, args with
       | "thread_idx_x", [_] -> Thread_idx_x
       | "load", [buffer; index] ->
           let index = expression env index in
           (match buffer.exp_desc with
            | Texp_ident { path=Path.Pident id; _ } ->
                (match identifier env (Path.Pident id) with Buffer i -> Load_f32 (i,index) | _ -> fail "load expects a buffer")
            | _ -> fail "load expects a buffer parameter")
       | "+.", [a;b] -> Add_f32 (expression env a, expression env b)
       | "*.", [a;b] -> Mul_f32 (expression env a, expression env b)
       | _ -> fail ("unsupported scalar expression " ^ name))
  | _ -> fail "unsupported expression in GPU kernel"

let rec statement env exp = match exp.exp_desc with
  | Texp_open (_, body) -> statement env body
  | Texp_let (Nonrecursive, [binding], body) ->
      let value = expression env binding.vb_expr in
      (match binding.vb_pat.pat_desc with
       | Tpat_var {id; _} -> statement ((id, Value value)::env) body
       | _ -> fail "kernel let binding must bind one name")
  | Texp_apply (callee, args, _, _, _, _) ->
      let name = match callee.exp_desc with
        | Texp_ident { path; _ } -> path_last path
        | _ -> fail "GPU statement target must be a resolved identifier" in
      let args = args_of_apply args in
      (match name, args with
       | "store", [buffer; index; value] ->
           let buffer = match buffer.exp_desc with
             | Texp_ident { path=Path.Pident id; _ } ->
                 (match identifier env (Path.Pident id) with Buffer i -> i | _ -> fail "store expects a buffer")
             | _ -> fail "store expects a buffer parameter" in
           let index = expression env index and value = expression env value in
           Printf.printf "body\tstore\t%d\t%s\t%s\n" buffer (emit_expr index) (emit_expr value)
       | _ -> fail ("unsupported GPU statement " ^ name))
  | _ -> fail "kernel body must end in a store"

let parameter_type typ = match Types.get_desc typ with
  | Types.Tconstr (path, _, _) -> Path.name path
  | _ -> ""

let last_component path = match List.rev (String.split_on_char '.' (Path.name path)) with
  | name :: _ -> name | [] -> ""

let rec kind_of_type typ = match Types.get_desc typ with
  | Types.Tpoly (inner, _) -> kind_of_type inner
  | Types.Tconstr (path, [element], _) when last_component path = "gpu_array" ->
      (match kind_of_type element with "scalar_f32" -> "buffer_f32" | _ -> "unsupported")
  | Types.Tconstr (path, [], _) ->
      (match last_component path with
       | "float" -> "scalar_f32" | "int" -> "scalar_i32" | "unit" -> "unit"
       | _ -> "unsupported")
  | _ -> "unsupported"

let emit_implementation_signature target binding =
  let name = match binding.vb_pat.pat_desc with
    | Tpat_var {name; _} -> name.txt
    | _ -> fail "kernel binding must have a simple name" in
  Printf.printf "kernel\t%s\n" name;
  let rec arrows index typ last_result_modes = match Types.get_desc typ with
    | Types.Tpoly (inner, _) -> arrows index inner last_result_modes
    | Types.Tarrow ((_, arg_modes, result_modes), arg, result, _) ->
        Printf.printf "arg\t%d\t%s\t%s\n" index (kind_of_type arg)
          (serialize_arrow_mode arg_modes);
        arrows (index + 1) result (Some result_modes)
    | _ ->
        (match last_result_modes with
         | Some modes ->
             Printf.printf "result\t%s\t%s\n" (kind_of_type typ)
               (serialize_arrow_mode modes)
         | None -> fail "typed implementation has no function result mode") in
  arrows 0 binding.vb_pat.pat_type None

let compile_kernel target structure =
  let binding = List.find_map (fun item -> match item.str_desc with
    | Tstr_value (_, bindings) -> List.find_opt (fun binding -> match binding.vb_pat.pat_desc with
        | Tpat_var {name; _} -> name.txt = target
        | _ -> false) bindings
    | _ -> None) structure.str_items in
  let binding = match binding with Some binding -> binding | None -> fail ("no kernel function named " ^ target) in
  emit_implementation_signature target binding;
  match binding.vb_expr.exp_desc with
  | Texp_function {params; body=Tfunction_body body; _} ->
      let env = List.mapi (fun index param ->
        let ty = match param.fp_kind with Tparam_pat pattern -> parameter_type pattern.pat_type | _ -> "" in
        if String.ends_with ~suffix:"gpu_array" ty then (param.fp_param, Buffer index)
        else (param.fp_param, Value (Arg index))) params in
      statement env body
  | _ -> fail "kernel must be a simple function with a body"

let () =
  if Array.length Sys.argv <> 3 then fail "usage: export_typedtree_modes FILE.cmt KERNEL_NAME";
  let _, cmt = Cmt_format.read Sys.argv.(1) in
  match cmt with
  | None -> fail "no OxCaml typedtree annotation found"
  | Some cmt -> match cmt.cmt_annots with
      | Implementation structure -> compile_kernel Sys.argv.(2) structure
      | _ -> fail "expected an OxCaml typed interface or implementation"
