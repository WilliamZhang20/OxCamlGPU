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

let gpu_array_path = persistent "Gpu_dsl" ["gpu_array"]
let gpu_shared_path = persistent "Gpu_dsl" ["gpu_shared"]
let gpu_mbarrier_path = persistent "Gpu_dsl" ["gpu_mbarrier"]
let tensor_map_path = persistent "Gpu_dsl" ["tensor_map"]
let wgmma_acc_path = persistent "Gpu_dsl" ["wgmma_acc"]

type kind =
  | Native_int
  | Int32
  | Int64
  | Float
  | Boolean
  | Buffer
  | Shared
  | Mbarrier
  | Tensor_map
  | Wgmma_acc
  | Unit

let path_uid env path =
  try Some (Env.find_type path env).type_uid with Not_found -> None

let rec kind env loc typ =
  let env = Envaux.env_of_only_summary env in
  let typ = Ctype.expand_head env typ in
  match Types.get_desc typ with
  | Types.Tpoly (t, _) -> kind env loc t
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_int -> Native_int
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_int32 -> Int32
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_int64 -> Int64
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_float -> Float
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_bool -> Boolean
  | Types.Tconstr (p, [], _) when Path.same p Predef.path_unit -> Unit
  | Types.Tconstr (p, [], _) ->
      (match path_uid env p
         , path_uid env gpu_mbarrier_path
         , path_uid env tensor_map_path
         , path_uid env wgmma_acc_path with
       | Some u, Some m, _, _ when Types.Uid.equal u m -> Mbarrier
       | Some u, _, Some t, _ when Types.Uid.equal u t -> Tensor_map
       | Some u, _, _, Some w when Types.Uid.equal u w -> Wgmma_acc
       | _ -> fail loc "unsupported GPU type declaration")
  | Types.Tconstr (p, [element], _) ->
      let uid = path_uid env p in
      let matches target =
        match uid, path_uid env target with
        | Some u, Some t -> Types.Uid.equal u t
        | _ -> false
      in
      if matches gpu_array_path && kind env loc element = Float then Buffer
      else if matches gpu_shared_path && kind env loc element = Float then Shared
      else fail loc "unsupported GPU type declaration"
  | _ -> fail loc "unsupported GPU source type"

let gpu_ty = function
  | Native_int | Int32 -> G.I32
  | Int64 -> G.U64
  | Float -> G.F32
  | Boolean -> G.Bool
  | Unit -> G.Unit
  | Buffer -> G.MemRef ([G.Dynamic], G.Float32, G.Global)
  | Shared -> G.MemRef ([G.Dynamic; G.Dynamic], G.Float32, G.Shared)
  | Mbarrier -> G.MemRef ([G.Static 2], G.Int32, G.Shared)
  | Tensor_map -> G.Tensor_map
  | Wgmma_acc -> G.F32 (* never a formal; adapter tracks Acc bindings *)

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

let gpu_ops =
  [ "thread_idx_x"; "global_idx_x"; "block_idx_x"; "block_idx_y"
  ; "load"; "store"; "load_masked"; "store_masked"
  ; "load_f32x4"; "store_f32x4"
  ; "shared"; "shared_wgmma"; "shared_load"; "shared_load_f32x4"; "shared_store"; "shared_store_f32x4"
  ; "barrier_cta"; "cp_async_f32x4"; "cp_async_commit"; "cp_async_wait"; "mad_f32"
  ; "warp_sum_f32"; "store_grid_leader"
  ; "warpgroup_index"; "grouped_tile_m"; "grouped_tile_n"
  ; "mbarrier"; "mbarrier_set"; "mbarrier_slot"; "mbarrier_init"; "mbarrier_init_elect"
  ; "mbarrier_arrive_expect_tx"; "mbarrier_arrive_expect_tx_elect"
  ; "mbarrier_arrive"; "mbarrier_arrive_elect"; "mbarrier_try_wait_parity"
  ; "tma_load_2d"; "tma_load_2d_elect"; "fence_proxy_async"
  ; "wgmma_fence"; "wgmma_acc"; "wgmma_acc_get"; "wgmma_acc_row"; "wgmma_acc_col"; "gmma_descriptor"; "wgmma_mma_tf32"
  ; "wgmma_commit_group"; "wgmma_wait_group"
  ]

let operations =
  List.map (fun n -> persistent "Gpu_dsl" ["Gpu"; n], n) gpu_ops @
  List.map (fun n -> persistent "Stdlib" [n], n)
    ["+."; "*."; "+"; "-"; "*"; "/"; "="; "<>"; "<"; "<="; ">"; ">=";
     "&&"; "||"; "not"] @
  List.map (fun n -> persistent "Stdlib" ["Int32"; n], "i32_" ^ n)
    ["add"; "sub"; "mul"; "div"; "rem"; "of_int"; "to_int"]

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
