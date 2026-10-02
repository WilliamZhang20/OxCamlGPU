(* Compiler-version-specific helpers for recognizing the supported OxCaml
   types, modes, and primitive declarations in Typedtree. *)
open Typedtree

module K = Kernel_ast
module G = Gpu_type
module M = Gpu_mode

let span (l : Location.t) =
  { Source_span.file = l.loc_start.pos_fname;
    start_line = l.loc_start.pos_lnum;
    start_col = l.loc_start.pos_cnum - l.loc_start.pos_bol;
    end_line = l.loc_end.pos_lnum;
    end_col = l.loc_end.pos_cnum - l.loc_end.pos_bol }

let fail loc text = Source_span.fail (span loc) text

let persistent root names =
  List.fold_left (fun p n -> Path.Pdot (p, n))
    (Path.Pident (Ident.create_persistent root)) names

let gpu_type_path = persistent "Gpu_dsl" ["gpu_array"]

type kind = Native_int | Int32 | Float | Boolean | Buffer | Unit

let rec kind env loc typ =
  let env = Envaux.env_of_only_summary env in
  let typ = Ctype.expand_head env typ in
  match Types.get_desc typ with
  | Types.Tpoly (t, _) -> kind env loc t
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_int -> Native_int
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_int32 -> Int32
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_float -> Float
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_bool -> Boolean
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_unit -> Unit
  | Types.Tconstr (p, [element], _) ->
      let matches = try
        Types.Uid.equal (Env.find_type p env).type_uid
          (Env.find_type gpu_type_path env).type_uid
      with Not_found -> false in
      if matches && kind env loc element = Float then Buffer
      else fail loc "unsupported GPU type declaration"
  | _ -> fail loc "unsupported GPU source type"

let gpu_ty = function
  | Native_int | Int32 -> G.I32
  | Float -> G.F32
  | Boolean -> G.Bool
  | Unit -> G.Unit
  | Buffer -> G.MemRef ([G.Dynamic], G.Float32, G.Global)

let slot env loc typ mode =
  let m = Mode.With_locality.to_const_exn mode in
  { K.loc = span loc;
    ty = gpu_ty (kind env loc typ);
    ownership = (match m.uniqueness with
      | Mode.Uniqueness.Const.Unique -> M.Unique
      | Aliased -> M.Aliased);
    locality = (match m.areality with
      | Mode.Locality.Const.Local -> M.Local
      | Global -> M.Global);
    domain_portability = (match m.portability with
      | Mode.Portability.Const.Portable -> M.Domain_portable
      | Nonportable -> M.Domain_nonportable
      | _ -> fail loc "unsupported shareable/corruptible portability");
    permission = (match m.visibility with
      | Mode.Visibility.Const.Read -> M.Read_only
      | Write -> M.Write_only
      | Read_write -> M.Read_write
      | Immutable -> M.Immutable);
    gpu_boundary = M.Boundary_unspecified }

let operations =
  List.map (fun n -> persistent "Gpu_dsl" ["Gpu"; n], n)
    ["thread_idx_x"; "global_idx_x"; "load"; "store"; "load_masked";
     "store_masked"; "warp_sum_f32"; "store_grid_leader"] @
  List.map (fun n -> persistent "Stdlib" [n], n)
    ["+."; "*."; "+"; "-"; "*"; "="; "<>"; "<"; "<="; ">"; ">=";
     "&&"; "||"; "not"] @
  List.map (fun n -> persistent "Stdlib" ["Int32"; n], "i32_" ^ n)
    ["add"; "sub"; "mul"; "of_int"; "to_int"]

let primitive exp =
  match exp.exp_desc with
  | Texp_ident { desc; _ } ->
      (match List.find_map (fun (path, name) ->
         let same = try
           let expected = Subst.Lazy.force_value_description
             (Env.find_value path exp.exp_env) in
           Types.Uid.equal expected.val_uid desc.val_uid
         with Not_found -> false in
         if same then Some name else None) operations with
       | Some name -> name
       | None -> fail exp.exp_loc
           "unsupported call declaration (only the GPU API and supported Stdlib primitives are accepted)")
  | _ -> fail exp.exp_loc "call target must be a resolved declaration"
