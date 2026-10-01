(* Version-matched OxCaml adapter. It reads compiler-owned Typedtree nodes and
   writes a versioned, flat operation stream; expressions are never serialized
   as a second textual language. *)
open Typedtree

let fail message = prerr_endline message; exit 2

let serialize_arrow_mode mode =
  let modes = Mode.With_locality.to_const_exn mode in
  let uniqueness = match modes.uniqueness with
    | Mode.Uniqueness.Const.Unique -> "unique"
    | Mode.Uniqueness.Const.Aliased -> "aliased" in
  let locality = match modes.areality with
    | Mode.Locality.Const.Local -> "local"
    | Mode.Locality.Const.Global -> "global" in
  let portability = match modes.portability with
    | Mode.Portability.Const.Portable -> "portable"
    | Mode.Portability.Const.Nonportable -> "nonportable"
    | Mode.Portability.Const.Shareable | Mode.Portability.Const.Corruptible ->
        fail "OxCaml shareable/corruptible portability is not represented by the GPU mode model" in
  let visibility = match modes.visibility with
    | Mode.Visibility.Const.Read_write -> "read_write"
    | Mode.Visibility.Const.Read -> "read"
    | Mode.Visibility.Const.Write -> "write"
    | Mode.Visibility.Const.Immutable -> "immutable" in
  String.concat "\t" [uniqueness; locality; portability; visibility]

type atom = Argument of int | Value of int
type memory = Buffer of int | Pointer of int
type binding = Scalar of atom | Memory of memory
type state = { mutable next_value : int }

let fresh state = let id = state.next_value in state.next_value <- id + 1; id
let atom = function Argument i -> "arg:" ^ string_of_int i | Value i -> "val:" ^ string_of_int i
let memory = function Buffer i -> "buf:" ^ string_of_int i | Pointer i -> "val:" ^ string_of_int i
let emit fields = Printf.printf "body\t%s\n" (String.concat "\t" fields)

let primitive_name path =
  let name = Path.name path in
  let primitive suffix short = if String.ends_with ~suffix name then Some short else None in
  match List.find_map Fun.id
    [ primitive "Gpu_dsl.Gpu.thread_idx_x" "thread_idx_x";
      primitive "Gpu_dsl.Gpu.global_idx_x" "global_idx_x";
      primitive "Gpu_dsl.Gpu.load" "load";
      primitive "Gpu_dsl.Gpu.load_masked" "load_masked";
      primitive "Gpu_dsl.Gpu.store" "store";
      primitive "Gpu_dsl.Gpu.store_masked" "store_masked";
      primitive "Gpu_dsl.Gpu.warp_sum_f32" "warp_sum_f32";
      primitive "Gpu_dsl.Gpu.store_grid_leader" "store_grid_leader";
      primitive ".+." "+.";
      primitive ".*." "*." ] with
  | Some short -> short
  | None -> fail ("unsupported call target in GPU kernel: " ^ name)

let identifier env path = match path with
  | Path.Pident id ->
      (match List.find_opt (fun (bound, _) -> Ident.same bound id) env with
       | Some (_, binding) -> binding
       | None -> fail ("unbound kernel value " ^ Ident.name id))
  | _ -> fail ("unsupported qualified value " ^ Path.name path)

let args_of_apply args = List.map (function
  | Nolabel, Typedtree.Arg (arg, _) -> arg
  | _ -> fail "only positional, fully applied GPU operations are supported") args

let rec scalar state env exp = match expression state env exp with
  | Scalar atom -> atom
  | Memory _ -> fail "buffer used as a scalar"

and buffer env exp = match exp.exp_desc with
  | Texp_ident { path; _ } ->
      (match identifier env path with
       | Memory (Buffer i) -> Buffer i
       | Memory (Pointer i) -> Pointer i
       | Scalar _ -> fail "scalar used as a buffer")
  | _ -> fail "buffer operand must be a named GPU memory value"

and expression state env exp = match exp.exp_desc with
  | Texp_constant (Const_int n) ->
      let id = fresh state in emit ["const_i32"; string_of_int id; string_of_int n]; Scalar (Value id)
  | Texp_constant (Const_float n) ->
      let id = fresh state in emit ["const_f32"; string_of_int id; n]; Scalar (Value id)
  | Texp_ident { path; _ } -> identifier env path
  | Texp_open (_, body) -> expression state env body
  | Texp_apply (callee, args, _, _, _, _) ->
      let name = match callee.exp_desc with
        | Texp_ident { path; _ } -> primitive_name path
        | _ -> fail "GPU call target must be a resolved identifier" in
      let args = args_of_apply args in
      (match name, args with
       | "thread_idx_x", [_] ->
           let id = fresh state in emit ["thread_idx_x"; string_of_int id]; Scalar (Value id)
       | "global_idx_x", [_] ->
           let id = fresh state in emit ["global_idx_x"; string_of_int id]; Scalar (Value id)
       | "load", [buffer_exp; index_exp] ->
           let base = buffer env buffer_exp in
           let index = scalar state env index_exp in
           let pointer = fresh state in
           emit ["gep_f32"; string_of_int pointer; memory base; atom index];
           let id = fresh state in emit ["load_f32"; string_of_int id; "val:" ^ string_of_int pointer];
           Scalar (Value id)
       | "load_masked", [buffer_exp; index_exp; bound_exp] ->
           let base = buffer env buffer_exp in
           let index = scalar state env index_exp in
           let bound = scalar state env bound_exp in
           let pointer = fresh state in
           emit ["gep_f32"; string_of_int pointer; memory base; atom index];
           let id = fresh state in
           emit ["load_f32_masked"; string_of_int id; "val:" ^ string_of_int pointer; atom index; atom bound];
           Scalar (Value id)
       | (("+." | "*.") as op), [a; b] ->
           let a = scalar state env a in
           let b = scalar state env b in
           let id = fresh state in
           let opcode = match op with "+." -> "add_f32" | _ -> "mul_f32" in
           emit [opcode; string_of_int id; atom a; atom b]; Scalar (Value id)
       | "warp_sum_f32", [value] ->
           let value = scalar state env value in
           let id = fresh state in emit ["warp_reduce_sum_f32"; string_of_int id; atom value];
           Scalar (Value id)
       | _ -> fail ("unsupported scalar expression " ^ name))
  | _ -> fail "unsupported expression in GPU kernel"

let rec statement state env exp = match exp.exp_desc with
  | Texp_open (_, body) -> statement state env body
  | Texp_sequence (first, _, second) ->
      statement state env first;
      statement state env second
  | Texp_let (Nonrecursive, [binding], body) ->
      let value = expression state env binding.vb_expr in
      (match binding.vb_pat.pat_desc with
       | Tpat_var { id; _ } -> statement state ((id, value) :: env) body
       | _ -> fail "kernel let binding must bind one name")
  | Texp_apply (callee, args, _, _, _, _) ->
      let name = match callee.exp_desc with
        | Texp_ident { path; _ } -> primitive_name path
        | _ -> fail "GPU statement target must be a resolved identifier" in
      let args = args_of_apply args in
      (match name, args with
       | "store", [buffer_exp; index_exp; value_exp] ->
           let base = buffer env buffer_exp in
           let index = scalar state env index_exp in
           let pointer = fresh state in emit ["gep_f32"; string_of_int pointer; memory base; atom index];
           let value = scalar state env value_exp in emit ["store_f32"; "val:" ^ string_of_int pointer; atom value]
       | "store_masked", [buffer_exp; index_exp; bound_exp; value_exp] ->
           let base = buffer env buffer_exp in
           let index = scalar state env index_exp in
           let bound = scalar state env bound_exp in
           let pointer = fresh state in emit ["gep_f32"; string_of_int pointer; memory base; atom index];
           let value = scalar state env value_exp in
           emit ["store_f32_masked"; "val:" ^ string_of_int pointer; atom value; atom index; atom bound]
       | "store_grid_leader", [buffer_exp; value_exp] ->
           let base = buffer env buffer_exp in
           let value = scalar state env value_exp in
           emit ["store_grid_leader_f32"; memory base; atom value]
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

let emit_implementation_signature binding =
  let name = match binding.vb_pat.pat_desc with
    | Tpat_var { name; _ } -> name.txt
    | _ -> fail "kernel binding must have a simple name" in
  Printf.printf "format\t2\nkernel\t%s\n" name;
  let rec arrows index typ last_result_modes = match Types.get_desc typ with
    | Types.Tpoly (inner, _) -> arrows index inner last_result_modes
    | Types.Tarrow ((_, arg_modes, result_modes), arg, result, _) ->
        Printf.printf "arg\t%d\t%s\t%s\n" index (kind_of_type arg) (serialize_arrow_mode arg_modes);
        arrows (index + 1) result (Some result_modes)
    | _ ->
        (match last_result_modes with
         | Some modes -> Printf.printf "result\t%s\t%s\n" (kind_of_type typ) (serialize_arrow_mode modes)
         | None -> fail "typed implementation has no function result mode") in
  arrows 0 binding.vb_pat.pat_type None

let compile_kernel target structure =
  let binding = List.find_map (fun item -> match item.str_desc with
    | Tstr_value (_, bindings) -> List.find_opt (fun binding -> match binding.vb_pat.pat_desc with
        | Tpat_var { name; _ } -> name.txt = target
        | _ -> false) bindings
    | _ -> None) structure.str_items in
  let binding = match binding with Some binding -> binding | None -> fail ("no kernel function named " ^ target) in
  emit_implementation_signature binding;
  match binding.vb_expr.exp_desc with
  | Texp_function { params; body=Tfunction_body body; _ } ->
      let env = List.mapi (fun index param ->
        let ty = match param.fp_kind with
          | Tparam_pat pattern -> parameter_type pattern.pat_type
          | _ -> fail "optional kernel parameters are unsupported" in
        let binding = match ty with
          | "Gpu_dsl.gpu_array" -> Memory (Buffer index)
          | "float" | "int" -> Scalar (Argument index)
          | _ -> fail ("unsupported kernel parameter type " ^ ty) in
        param.fp_param, binding) params in
      statement { next_value=0 } env body
  | _ -> fail "kernel must be a simple function with a body"

let () =
  if Array.length Sys.argv <> 3 then fail "usage: export_typedtree_modes FILE.cmt KERNEL_NAME";
  let _, cmt = Cmt_format.read Sys.argv.(1) in
  match cmt with
  | None -> fail "no OxCaml typedtree annotation found"
  | Some cmt -> match cmt.cmt_annots with
      | Implementation structure -> compile_kernel Sys.argv.(2) structure
      | _ -> fail "expected an OxCaml typed interface or implementation"
