open Gpu_type
open Gpu_mode
open Ir

type error = Verification_error.error = { code : string; message : string }
exception Invalid_kernel of error list

let tile_type = function
  | Tensor (shape, dtype) -> Some (shape, dtype)
  | _ -> None

let memref_type = function
  | MemRef (shape, dtype, space) -> Some (shape, dtype, space)
  | _ -> None

let is_static_shape shape =
  List.for_all (function Static n -> n > 0 | _ -> false) shape

let readable = can_read
let writable = can_write

let may_alias = Alias_analysis.may_alias

let verify_kernel k =
  let errors = ref [] in
  let current_loc = ref Source_span.synthetic in
  let add code message =
    errors :=
      { code; message = Source_span.to_string !current_loc ^ ": " ^ message }
      :: !errors
  in
  let check_layout value =
    let validate expected shape layout =
      (match layout with
       | layout when expected layout ->
           (match Layout.validate shape layout with
            | Ok () -> ()
            | Error message -> add "E_LAYOUT_INVALID" (Printf.sprintf "value %d: %s" value.id message))
       | _ -> add "E_LAYOUT_KIND" (Printf.sprintf "value %d has a layout incompatible with its type" value.id)) in
    match value.ty, value.layout with
    | Tensor (shape, _), Some ((Layout.Register _ | Layout.Mapped_register _) as layout) ->
        validate (function Layout.Register _ | Layout.Mapped_register _ -> true | _ -> false) shape layout
    | Tensor _, _ -> add "E_TENSOR_LAYOUT" (Printf.sprintf "tensor value %d requires a register layout" value.id)
    | MemRef (shape, _, Shared), Some ((Layout.Shared _ | Layout.Padded_shared _) as layout) ->
        validate (function Layout.Shared _ | Layout.Padded_shared _ -> true | _ -> false) shape layout
    | MemRef (_, _, Shared), _ -> add "E_SHARED_LAYOUT" (Printf.sprintf "shared MemRef %d requires a shared layout" value.id)
    | MemRef (shape, _, Global), Some (Layout.Global _ as layout) ->
        validate (function Layout.Global _ -> true | _ -> false) shape layout
    | MemRef (_, _, (Global | Local)), Some _ ->
        add "E_MEMREF_LAYOUT" "global MemRefs may only carry an explicit Global stride layout"
    | _, Some _ -> add "E_LAYOUT_KIND" (Printf.sprintf "scalar/pointer value %d cannot carry a tile layout" value.id)
    | _, None -> () in
  let arg_ids = List.map (fun a -> a.value.id) k.args in
  List.iter (fun a ->
    current_loc := a.value.loc;
    check_layout a.value;
    if a.value.id < 0 then add "E_DUPLICATE_ARG_ID" "argument ids must be nonnegative";
    match a.value.ty with
    | Ptr (F32, Global) | F32 | I32 | Bool | Tensor_map -> ()
    | ty when is_global_f32_memref ty || is_static_global_f32_tile ty -> ()
    | _ -> add "E_ARG_TYPE" ("argument " ^ a.name ^ " has an unsupported GPU kernel ABI type")) k.args;
  let ids = List.sort compare arg_ids in
  let rec duplicate = function x :: y :: _ when x = y -> true | _ :: tl -> duplicate tl | [] -> false in
  if duplicate ids then add "E_DUPLICATE_ARG_ID" "argument value ids must be distinct";
  let seen = Hashtbl.create 16 in
  let gep_indices = Hashtbl.create 16 in
  List.iter (fun a -> Hashtbl.replace seen a.value.id a.value) k.args;
  let all_ids = Hashtbl.copy seen in
  let check_use seen v =
    match Hashtbl.find_opt seen v.id with
    | None -> add "E_UNDEFINED" (Printf.sprintf "value %d is used before definition or outside its scope" v.id)
    | Some canonical when canonical <> v -> add "E_VALUE_IDENTITY" "operand disagrees with canonical SSA definition"
    | Some _ -> () in
  let require_bool pred what =
    Option.iter (fun p ->
      if p.ty <> Bool then add "E_CONDITION_TYPE" (what ^ " pred must be bool")) pred
  in
  let check_mbarrier_memref mbar ~writable_needed ~readable_needed what =
    match memref_type mbar.ty with
    | Some ([Static 2], Int32, Shared) ->
        if writable_needed && not (writable mbar.permission) then
          add "E_READONLY_STORE" (what ^ " requires a writable shared MemRef");
        if readable_needed && not (readable mbar.permission) then
          add "E_UNREADABLE_LOAD" (what ^ " requires a readable shared MemRef");
        true
    | _ -> false
  in
  let rec visit seen nested body =
  List.iter (fun i ->
    current_loc := (match results i @ operands i with
      | v :: _ -> v.loc
      | [] -> Source_span.synthetic);
    List.iter check_layout (operands i @ results i);
    List.iter (check_use seen) (operands i);
    (match i with
    (* --- scalars / control --- *)
    | Const_bool(dst,_) -> if dst.ty <> Bool then add "E_CONST_TYPE" "bool constant must define bool"
    | Compare(dst,cmp,a,b) ->
        if dst.ty<>Bool || a.ty<>b.ty || not (List.mem a.ty [I32;F32;Bool]) ||
           (a.ty=Bool && not (List.mem cmp [Eq;Ne])) then add "E_COMPARE_TYPE" "comparison operand/result mismatch"
    | If(dst,c,a,b) | If_uni(dst,c,a,b) ->
        if c.ty <> Bool then add "E_CONDITION_TYPE" "if condition must be bool";
        (match dst with
         | Some d when not (List.mem d.ty [I32; F32; Bool]) ->
             add "E_YIELD_TYPE" "branch results must be scalar"
         | _ -> ());
        let branch_loc = !current_loc in
        let branch (r : Ir.region) =
          let scope = Hashtbl.copy seen in visit scope true r.body;
          current_loc := branch_loc;
          Option.iter (check_use scope) r.yield;
          match dst,r.yield with
          | None,None -> ()
          | Some d, Some v when d.ty = v.ty ->
              if d.ownership <> Aliased || d.provenance <> None ||
                 not (locality_satisfies ~required:d.locality v.locality) ||
                 not (domain_portability_satisfies ~required:d.domain_portability v.domain_portability) ||
                 not (gpu_boundary_satisfies ~required:d.gpu_boundary v.gpu_boundary) ||
                 not (satisfies ~required:d.permission v.permission) then
                add "E_YIELD_FACTS" "branch result strengthens yielded modal facts"
          | _ -> add "E_YIELD_TYPE" "branch yield must match declared if result" in
        branch a; branch b
    | For (induction, limit, step, body) ->
        if induction.ty <> I32 || limit.ty <> I32 then
          add "E_FOR_TYPE" "for induction and limit must be i32";
        if step <= 0 then
          add "E_FOR_STEP" "for step must be positive";
        let loop_loc = !current_loc in
        let scope = Hashtbl.copy seen in
        visit scope true body;
        current_loc := loop_loc
    | Const_i32 (dst, n) ->
        if Int64.of_int n < -2147483648L || Int64.of_int n > 2147483647L then add "E_CONST_RANGE" "i32 literal is out of range";
        if dst.ty <> I32 then add "E_CONST_TYPE" "i32 constant must define an i32 value"
    | Const_f32 (dst, _) ->
        if dst.ty <> F32 then add "E_CONST_TYPE" "f32 constant must define an f32 value"
    | Thread_idx_x dst | Global_idx_x dst | Block_idx_x dst | Block_idx_y dst ->
        if dst.ty <> I32 then add "E_INDEX_TYPE" "thread and block indices must define i32 values"
    | Add_i32 (dst, a, b) | Sub_i32(dst,a,b) | Mul_i32(dst,a,b)
    | Div_i32(dst,a,b) | Rem_i32(dst,a,b) ->
        if dst.ty <> I32 || a.ty <> I32 || b.ty <> I32 then
          add "E_ARITH_TYPE" "i32 arithmetic requires an i32 result and i32 operands"
    | Add_f32 (dst, a, b) ->
        if dst.ty <> F32 || a.ty <> F32 || b.ty <> F32 then
          add "E_ARITH_TYPE" "f32 add requires an f32 result and f32 operands"
    | Mul_f32 (dst, a, b) ->
        if dst.ty <> F32 || a.ty <> F32 || b.ty <> F32 then
          add "E_ARITH_TYPE" "f32 multiply requires an f32 result and f32 operands"
    | Mad_f32 (dst, a, b, c) ->
        if dst.ty <> F32 || a.ty <> F32 || b.ty <> F32 || c.ty <> F32 then
          add "E_ARITH_TYPE" "mad_f32 requires an f32 result and f32 operands"
    (* --- global memory --- *)
    | Store_f32 (ptr, value) ->
        (match ptr.ty, value.ty with
         | Ptr (F32, Global), F32 -> ()
         | _ -> add "E_STORE_TYPE" "store requires a global f32 pointer and f32 value");
        if not (writable ptr.permission) then
          add "E_READONLY_STORE" (Printf.sprintf "store through non-writable value %d" ptr.id)
    | Store_f32_grid_leader (ptr, value) ->
        (match ptr.ty, value.ty with
         | (Ptr (F32, Global) | MemRef (_, Float32, Global)), F32 -> ()
         | _ -> add "E_STORE_TYPE" "grid-leader store requires a global f32 buffer and f32 value");
        if not (writable ptr.permission) then
          add "E_READONLY_STORE" (Printf.sprintf "grid-leader store through non-writable value %d" ptr.id)
    | Load_f32_masked (dst, ptr, ix, bound) ->
        (match dst.ty, ptr.ty, ix.ty, bound.ty with
         | F32, Ptr (F32, Global), I32, I32 -> ()
         | _ -> add "E_LOAD_TYPE" "masked load requires f32, global f32 pointer, and i32 index/bound");
        if not (readable ptr.permission) then
          add "E_UNREADABLE_LOAD" (Printf.sprintf "masked load through non-readable value %d" ptr.id);
        if Hashtbl.find_opt gep_indices ptr.id <> Some ix.id then
          add "E_MASK_INDEX" "masked load bound must guard the index used to form its pointer"
    | Store_f32_masked (ptr, value, ix, bound) ->
        (match ptr.ty, value.ty, ix.ty, bound.ty with
         | Ptr (F32, Global), F32, I32, I32 -> ()
         | _ -> add "E_STORE_TYPE" "masked store requires global f32 pointer, f32 value, and i32 index/bound");
        if not (writable ptr.permission) then
          add "E_READONLY_STORE" (Printf.sprintf "masked store through non-writable value %d" ptr.id);
        if Hashtbl.find_opt gep_indices ptr.id <> Some ix.id then
          add "E_MASK_INDEX" "masked store bound must guard the index used to form its pointer"
    | Load_f32 (dst, ptr) ->
        (match dst.ty, ptr.ty with
         | F32, Ptr (F32, Global) -> ()
         | _ -> add "E_LOAD_TYPE" "load requires an f32 result and global f32 pointer");
        if not (readable ptr.permission) then
          add "E_UNREADABLE_LOAD" (Printf.sprintf "load through non-readable value %d" ptr.id)
    | Load_f32x4 (a, b, c, d, ptr) ->
        (match a.ty, b.ty, c.ty, d.ty, ptr.ty with
         | F32, F32, F32, F32, Ptr (F32, Global) -> ()
         | _ -> add "E_LOAD_TYPE" "f32x4 load requires four f32 results and a global f32 pointer");
        if not (readable ptr.permission) then
          add "E_UNREADABLE_LOAD" (Printf.sprintf "f32x4 load through non-readable value %d" ptr.id)
    | Store_f32x2 (ptr, a, b) ->
        (match ptr.ty, a.ty, b.ty with
         | Ptr (F32, Global), F32, F32 -> ()
         | _ -> add "E_STORE_TYPE" "f32x2 store requires a global f32 pointer and two f32 values");
        if not (writable ptr.permission) then
          add "E_READONLY_STORE" (Printf.sprintf "f32x2 store through non-writable value %d" ptr.id)
    | Store_f32x4 (ptr, a, b, c, d) ->
        (match ptr.ty, a.ty, b.ty, c.ty, d.ty with
         | Ptr (F32, Global), F32, F32, F32, F32 -> ()
         | _ -> add "E_STORE_TYPE" "f32x4 store requires a global f32 pointer and four f32 values");
        if not (writable ptr.permission) then
          add "E_READONLY_STORE" (Printf.sprintf "f32x4 store through non-writable value %d" ptr.id)
    | Gep_f32 (dst, ptr, ix) ->
        Hashtbl.replace gep_indices dst.id ix.id;
        if dst.ty <> Ptr (F32, Global) || not (is_global_f32_buffer ptr.ty) || ix.ty <> I32 then
          add "E_GEP_TYPE" "f32 element address requires global f32 pointers and i32 index";
        if dst.permission <> ptr.permission || dst.ownership <> Aliased ||
           dst.provenance <> ptr.provenance ||
           dst.domain_portability <> ptr.domain_portability || dst.gpu_boundary <> ptr.gpu_boundary ||
           (ptr.locality = Local && dst.locality <> Local) then
          add "E_GEP_FACTS" "element address must be aliased, retain root provenance/access facts, and cannot widen locality"
    (* --- shared memory --- *)
    | Shared_alloc memref ->
        (match memref_type memref.ty with
         | Some (shape, _, Shared) ->
             if not (is_static_shape shape) then
               add "E_SHARED_DYNAMIC_ALLOC" "shared allocation requires a fully static shape"
         | _ -> add "E_SHARED_ALLOC_TYPE" "shared allocation must define a shared-space MemRef");
        (match memref.permission with
         | Read_only | Immutable -> add "E_SHARED_ALLOC_PERMISSION" "shared allocation must be writable"
         | Write_only | Read_write -> ())
    | Shared_load_f32 (dst, memref, index) ->
        (match dst.ty, memref_type memref.ty, index.ty with
         | F32, Some (_, Float32, Shared), I32 ->
             (match memref.layout with
              | Some (Layout.Shared _ | Layout.Padded_shared _) -> ()
              | _ -> add "E_SHARED_LAYOUT" "shared load requires a shared MemRef layout");
             if not (readable memref.permission) then
               add "E_UNREADABLE_LOAD" (Printf.sprintf "shared load through non-readable MemRef %d" memref.id)
         | _ -> add "E_LOAD_TYPE" "shared load requires f32 result, shared f32 MemRef, and i32 index")
    | Shared_load_f32x4 (a, b, c, d, memref, index) ->
        (match a.ty, b.ty, c.ty, d.ty, memref_type memref.ty, index.ty with
         | F32, F32, F32, F32, Some (_, Float32, Shared), I32 ->
             (match memref.layout with
              | Some (Layout.Shared _ | Layout.Padded_shared _) -> ()
              | _ -> add "E_SHARED_LAYOUT" "shared f32x4 load requires a shared MemRef layout");
             if not (readable memref.permission) then
               add "E_UNREADABLE_LOAD"
                 (Printf.sprintf "shared f32x4 load through non-readable MemRef %d" memref.id)
         | _ ->
             add "E_LOAD_TYPE"
               "shared f32x4 load requires four f32 results, shared f32 MemRef, and i32 index")
    | Shared_store_f32 (memref, value, index) ->
        (match memref_type memref.ty, value.ty, index.ty with
         | Some (_, Float32, Shared), F32, I32 ->
             (match memref.layout with
              | Some (Layout.Shared _ | Layout.Padded_shared _) -> ()
              | _ -> add "E_SHARED_LAYOUT" "shared store requires a shared MemRef layout");
             if not (writable memref.permission) then
               add "E_READONLY_STORE" (Printf.sprintf "shared store through non-writable MemRef %d" memref.id)
         | _ -> add "E_STORE_TYPE" "shared store requires shared f32 MemRef, f32 value, and i32 index")
    | Shared_store_f32x4 (memref, a, b, c, d, index) ->
        (match memref_type memref.ty, a.ty, b.ty, c.ty, d.ty, index.ty with
         | Some (_, Float32, Shared), F32, F32, F32, F32, I32 ->
             (match memref.layout with
              | Some (Layout.Shared _ | Layout.Padded_shared _) -> ()
              | _ -> add "E_SHARED_LAYOUT" "shared f32x4 store requires a shared MemRef layout");
             if not (writable memref.permission) then
               add "E_READONLY_STORE"
                 (Printf.sprintf "shared f32x4 store through non-writable MemRef %d" memref.id)
         | _ ->
             add "E_STORE_TYPE"
               "shared f32x4 store requires shared f32 MemRef, four f32 values, and i32 index")
    (* --- async / mbarrier / TMA / WGMMA --- *)
    | Cp_async_shared_f32x4 (memref, index, ptr) ->
        (match memref_type memref.ty, index.ty, ptr.ty with
         | Some (_, Float32, Shared), I32, Ptr (F32, Global) ->
             (match memref.layout with
              | Some (Layout.Shared _ | Layout.Padded_shared _) -> ()
              | _ -> add "E_SHARED_LAYOUT" "cp.async requires a shared MemRef layout");
             if not (writable memref.permission) then
               add "E_READONLY_STORE" "cp.async destination must be writable";
             if not (readable ptr.permission) then
               add "E_UNREADABLE_LOAD" "cp.async source must be readable"
         | _ ->
             add "E_ASYNC_TYPE"
               "cp.async requires shared f32 MemRef, i32 index, and global f32 pointer")
    | Cp_async_commit | Cp_async_wait_all | Fence_proxy_async
    | Wgmma_fence | Wgmma_commit_group | Wgmma_wait_group _ -> ()
    | Mbarrier_init (mbar, count, pred) ->
        require_bool pred "mbarrier";
        if count.ty <> I32
           || not (check_mbarrier_memref mbar ~writable_needed:true ~readable_needed:false
                     "mbarrier init") then
          add "E_MBARRIER_TYPE"
            "mbarrier init requires shared i32[2] MemRef (8B) and i32 arrive count"
        else
          (match mbar.layout with
           | Some (Layout.Shared _) -> ()
           | _ -> add "E_SHARED_LAYOUT" "mbarrier requires a shared MemRef layout")
    | Mbarrier_arrive_expect_tx (mbar, tx, pred) ->
        require_bool pred "mbarrier";
        if tx.ty <> I32
           || not (check_mbarrier_memref mbar ~writable_needed:true ~readable_needed:false
                     "mbarrier arrive.expect_tx") then
          add "E_MBARRIER_TYPE"
            "mbarrier arrive.expect_tx requires shared i32[2] MemRef and i32 tx bytes"
    | Mbarrier_arrive (mbar, pred) ->
        require_bool pred "mbarrier";
        if not (check_mbarrier_memref mbar ~writable_needed:true ~readable_needed:false
                  "mbarrier arrive") then
          add "E_MBARRIER_TYPE" "mbarrier arrive requires shared i32[2] MemRef"
    | Mbarrier_slot (dst, set, index) ->
        (match memref_type dst.ty, memref_type set.ty, index.ty with
         | Some ([Static 2], Int32, Shared), Some ([Static n], Int32, Shared), I32
           when n >= 2 && n mod 2 = 0 -> ()
         | _ ->
             add "E_MBARRIER_SLOT_TYPE"
               "an mbarrier slot needs a shared mbarrier set and an i32 index");
        if not (writable set.permission) then
          add "E_MBARRIER_SLOT_PERMISSION"
            (Printf.sprintf "mbarrier set %d is not writable" set.id)
    | Mbarrier_try_wait_parity (mbar, parity) ->
        if parity.ty <> I32
           || not (check_mbarrier_memref mbar ~writable_needed:false ~readable_needed:true
                     "mbarrier try_wait") then
          add "E_MBARRIER_TYPE"
            "mbarrier try_wait.parity requires shared i32[2] MemRef and i32 parity"
    | Tma_load_2d (smem, index, tmap, c0, c1, mbar, pred) ->
        Option.iter (fun p -> if p.ty <> Bool then add "E_CONDITION_TYPE" "TMA pred must be bool") pred;
        (match memref_type smem.ty, index.ty, tmap.ty, c0.ty, c1.ty, memref_type mbar.ty with
         | Some (_, Float32, Shared), I32, Tensor_map, I32, I32, Some ([Static 2], Int32, Shared) ->
             (match smem.layout with
              | Some (Layout.Shared _ | Layout.Padded_shared _) -> ()
              | _ -> add "E_SHARED_LAYOUT" "TMA destination requires a shared MemRef layout");
             if not (writable smem.permission) then
               add "E_READONLY_STORE" "TMA destination must be writable";
             if not (readable tmap.permission) then
               add "E_UNREADABLE_LOAD" "TMA tensor map must be readable";
             if not (writable mbar.permission) then
               add "E_READONLY_STORE" "TMA mbarrier must be writable"
         | _ ->
             add "E_TMA_TYPE"
               "TMA load requires shared f32 MemRef, i32 index, Tensor_map, i32 coords, and shared i32[2] mbarrier")
    | Wgmma_mma_tf32 (acc, desc_a, desc_b, n, _) ->
        let n_ok = n >= 8 && n <= 256 && n mod 8 = 0 in
        let expect = n / 2 in
        if not n_ok then
          add "E_WGMMA_SHAPE" "WGMMA n must be in {8,16,...,256}";
        if List.length acc <> expect then
          add "E_WGMMA_ACC"
            (Printf.sprintf "WGMMA m64n%dk8 expects %d f32 acc regs per thread" n expect);
        List.iter (fun v ->
          if v.ty <> F32 then add "E_WGMMA_ACC" "WGMMA accumulator registers must be f32") acc;
        if desc_a.ty <> U64 || desc_b.ty <> U64 then
          add "E_WGMMA_TYPE" "WGMMA descriptors must be u64 (from Gmma_descriptor)"
    | Gmma_descriptor (dst, smem, index, fields) ->
        if dst.ty <> U64 then
          add "E_GMMA_DESC" "Gmma_descriptor destination must be u64";
        if index.ty <> I32 then
          add "E_GMMA_DESC" "Gmma_descriptor index must be i32";
        if fields.leading < 0 || fields.stride < 0
           || fields.layout_type < 0 || fields.layout_type > 3 then
          add "E_GMMA_DESC" "Gmma_descriptor fields out of range";
        (match memref_type smem.ty with
         | Some (_, Float32, Shared) ->
             if not (readable smem.permission) then
               add "E_UNREADABLE_LOAD" "Gmma_descriptor shared operand must be readable"
         | _ ->
             add "E_GMMA_DESC" "Gmma_descriptor requires a shared f32 MemRef")
    (* --- tensors --- *)
    | Load_tensor (dst, src) ->
        (match tile_type dst.ty, memref_type src.ty with
         | Some (dst_shape, dst_dtype), Some (src_shape, src_dtype, space) ->
             if dst_shape <> src_shape || dst_dtype <> src_dtype then
               add "E_TILE_TRANSFER_TYPE" "tile load source and destination must have matching shape and dtype";
             if not (readable src.permission) then
               add "E_UNREADABLE_LOAD" (Printf.sprintf "tile load through non-readable MemRef %d" src.id);
             (match space, src.layout with
              | Shared, Some (Layout.Shared _ | Layout.Padded_shared _) -> ()
              | Shared, _ -> add "E_SHARED_LAYOUT" "shared tile source requires a shared layout"
              | Global, (None | Some (Layout.Global _)) -> ()
              | Local, _ -> ()
              | Global, Some _ -> add "E_MEMREF_LAYOUT" "global tile source has an unsupported layout")
         | _ -> add "E_TILE_LOAD_TYPE" "tile load requires a Tensor destination and MemRef source");
        (match dst.layout with Some (Layout.Register _ | Layout.Mapped_register _) -> () | _ -> add "E_TENSOR_LAYOUT" "tile load destination requires a register layout")
    | Load_tensor_masked (dst, src, rows, cols) ->
        (match tile_type dst.ty, memref_type src.ty, rows.ty, cols.ty with
         | Some (dst_shape, dst_dtype), Some (src_shape, src_dtype, space), I32, I32 ->
             if List.length dst_shape <> 2 || dst_shape <> src_shape || dst_dtype <> src_dtype then
               add "E_TILE_TRANSFER_TYPE" "masked tile load requires matching rank-2 tensors and MemRefs";
             if not (readable src.permission) then
               add "E_UNREADABLE_LOAD" (Printf.sprintf "masked tile load through non-readable MemRef %d" src.id);
             (match space, src.layout with
              | Global, (None | Some (Layout.Global _)) -> ()
              | Shared, Some (Layout.Shared { vector_width=1; swizzle=Layout.No_swizzle; _ }) -> ()
              | _ -> add "E_TILE_LOAD_TYPE" "masked tile load supports global or plain shared MemRefs")
         | _ -> add "E_TILE_LOAD_TYPE" "masked tile load requires tensor, memref, and i32 row/col bounds");
        (match dst.layout with Some (Layout.Register _ | Layout.Mapped_register _) -> () | _ -> add "E_TENSOR_LAYOUT" "masked tile load destination requires a register layout")
    | Store_tensor (dst, src) ->
        (match memref_type dst.ty, tile_type src.ty with
         | Some (dst_shape, dst_dtype, space), Some (src_shape, src_dtype) ->
             if dst_shape <> src_shape || dst_dtype <> src_dtype then
               add "E_TILE_TRANSFER_TYPE" "tile store source and destination must have matching shape and dtype";
             if not (writable dst.permission) then
               add "E_READONLY_STORE" (Printf.sprintf "tile store through non-writable MemRef %d" dst.id);
             (match space, dst.layout with
              | Shared, Some (Layout.Shared _ | Layout.Padded_shared _) -> ()
              | Shared, _ -> add "E_SHARED_LAYOUT" "shared tile destination requires a shared layout"
              | Global, (None | Some (Layout.Global _)) -> ()
              | Local, _ -> ()
              | Global, Some _ -> add "E_MEMREF_LAYOUT" "global tile destination has an unsupported layout")
         | _ -> add "E_TILE_STORE_TYPE" "tile store requires a MemRef destination and Tensor source");
        (match src.layout with Some (Layout.Register _ | Layout.Mapped_register _) -> () | _ -> add "E_TENSOR_LAYOUT" "tile store source requires a register layout")
    | Store_tensor_masked (dst, src, rows, cols) ->
        (match memref_type dst.ty, tile_type src.ty, rows.ty, cols.ty with
         | Some (dst_shape, dst_dtype, space), Some (src_shape, src_dtype), I32, I32 ->
             if List.length dst_shape <> 2 || dst_shape <> src_shape || dst_dtype <> src_dtype then
               add "E_TILE_TRANSFER_TYPE" "masked tile store requires matching rank-2 tensors and MemRefs";
             if not (writable dst.permission) then
               add "E_READONLY_STORE" (Printf.sprintf "masked tile store through non-writable MemRef %d" dst.id);
             (match space, dst.layout with
              | Global, (None | Some (Layout.Global _)) -> ()
              | Shared, Some (Layout.Shared { vector_width=1; swizzle=Layout.No_swizzle; _ }) -> ()
              | _ -> add "E_TILE_STORE_TYPE" "masked tile store supports global or plain shared MemRefs")
         | _ -> add "E_TILE_STORE_TYPE" "masked tile store requires memref, tensor, and i32 row/col bounds");
        (match src.layout with Some (Layout.Register _ | Layout.Mapped_register _) -> () | _ -> add "E_TENSOR_LAYOUT" "masked tile store source requires a register layout")
    | Remap_tensor (dst, src) ->
        (match dst.ty, src.ty with
         | Tensor (dshape, Float32), Tensor (sshape, Float32) when dshape=sshape -> ()
         | _ -> add "E_REMAP_TYPE" "tensor remap requires matching f32 tensor shapes");
        (match dst.layout, src.layout with
         | Some (Layout.Register _ | Layout.Mapped_register _),
           Some (Layout.Register _ | Layout.Mapped_register _) ->
             if dst.layout = src.layout then
               add "E_REMAP_LAYOUT" "tensor remap source and destination layouts must differ"
         | _ -> add "E_TENSOR_LAYOUT" "tensor remap requires register layouts on both sides")
    | Scale_tensor_f32 (dst, src, scalar) ->
        if dst.ty <> src.ty ||
           (match dst.ty with Tensor (_, Float32) -> false | _ -> true) ||
           scalar.ty <> F32 then
          add "E_TILE_ARITH_TYPE" "f32 tile scaling requires matching Tensor values and an f32 scalar";
        (match dst.layout, src.layout with
         | Some (Layout.Register _ | Layout.Mapped_register _), Some (Layout.Register _ | Layout.Mapped_register _)
           when dst.layout = src.layout -> ()
         | _ -> add "E_TENSOR_LAYOUT" "tile scaling requires identical register layouts on input and output")
    | Mul_tensor_f32 (dst, a, b) ->
        if dst.ty <> a.ty || dst.ty <> b.ty ||
           (match dst.ty with Tensor (_, Float32) -> false | _ -> true) then
          add "E_TILE_ARITH_TYPE" "f32 tensor multiplication requires matching f32 Tensor values";
        (match dst.layout, a.layout, b.layout with
         | Some (Layout.Register _ | Layout.Mapped_register _), Some (Layout.Register _ | Layout.Mapped_register _), Some (Layout.Register _ | Layout.Mapped_register _)
           when dst.layout = a.layout && a.layout = b.layout -> ()
         | _ -> add "E_TENSOR_LAYOUT" "tensor multiplication requires identical register layouts")
    | Reduce_sum_f32 (dst, src) ->
        (match dst.ty, src.ty with
         | F32, Tensor (_, Float32) -> ()
         | _ -> add "E_REDUCE_TYPE" "f32 sum requires an f32 Tensor and returns f32");
        (match src.layout with
         | Some (Layout.Register _ | Layout.Mapped_register _) -> ()
         | _ -> add "E_TENSOR_LAYOUT" "reduction input requires a register layout")
    (* --- collectives / returns --- *)
    | Warp_sum_f32 (dst, src) ->
        if dst.ty <> F32 || src.ty <> F32 then
          add "E_WARP_REDUCE_TYPE" "warp sum currently reduces f32 values and returns f32"
    | Barrier _ -> ()
    | Return None when nested -> add "E_REGION_RETURN" "branch regions must yield, not return"
    | Return (Some _) -> add "E_KERNEL_RETURN_TYPE" "GPU kernel entry points must return unit"
    | _ -> ());
    List.iter (fun v ->
      if v.id < 0 || Hashtbl.mem all_ids v.id then
        add "E_REDEFINITION" "invalid or duplicate SSA definition"
      else (Hashtbl.add all_ids v.id v; Hashtbl.add seen v.id v))
      (results i)) body
  in
  visit seen false k.body;
  (* Definite initialization intersects at joins; pending writes union. *)
  let rec shared initialized pending body =
    List.iter (function
      | Shared_alloc m -> Hashtbl.replace initialized m.id false
      | Store_tensor (m, _) | Store_tensor_masked (m, _, _, _)
        | Shared_store_f32 (m, _, _) | Shared_store_f32x4 (m, _, _, _, _, _)
        when (match m.ty with MemRef (_, _, Shared) -> true | _ -> false) ->
          Hashtbl.replace initialized m.id true; Hashtbl.replace pending m.id true
      | Cp_async_shared_f32x4 (m, _, _) | Tma_load_2d (m, _, _, _, _, _, _)
        when (match m.ty with MemRef (_, _, Shared) -> true | _ -> false) ->
          (* Initialize only: double-buffered schedules overlap a copy into
             stage 1 with reads of stage 0 in the same MemRef, so pending would
             false-positive. CTA barriers / mbarrier+fence are the sync points. *)
          Hashtbl.replace initialized m.id true
      | Mbarrier_try_wait_parity _ | Fence_proxy_async ->
          (* TMA completion + proxy fence make shared tiles visible. *)
          Hashtbl.clear pending
      | Load_tensor (_, m) | Load_tensor_masked (_, m, _, _)
        | Shared_load_f32 (_, m, _) | Shared_load_f32x4 (_, _, _, _, m, _)
        when (match m.ty with MemRef (_, _, Shared) -> true | _ -> false) ->
          if Hashtbl.find_opt initialized m.id <> Some true then add "E_SHARED_UNINITIALIZED" "shared tile is not definitely initialized"
          else if Hashtbl.mem pending m.id then add "E_SHARED_SYNC" "shared tile is read before a CTA barrier"
      | Barrier Execution.Cta -> Hashtbl.clear pending
      | If (_, _, a, b) | If_uni (_, _, a, b) ->
          (* SPMD role splits (producer/consumer) both execute: a write in either
             arm initializes the tile for the CTA after a following barrier.
             Use OR for initialized; pending is still the union. *)
          let ai = Hashtbl.copy initialized and ap = Hashtbl.copy pending in
          let bi = Hashtbl.copy initialized and bp = Hashtbl.copy pending in
          shared ai ap a.body; shared bi bp b.body;
          let mark id =
            Hashtbl.replace initialized id
              (Hashtbl.find_opt ai id = Some true
               || Hashtbl.find_opt bi id = Some true)
          in
          Hashtbl.iter (fun id _ -> mark id) ai;
          Hashtbl.iter (fun id _ -> mark id) bi;
          Hashtbl.clear pending;
          Hashtbl.iter (fun id _ -> Hashtbl.replace pending id true) ap;
          Hashtbl.iter (fun id _ -> Hashtbl.replace pending id true) bp
      | For (_, _, _, loop_body) -> shared initialized pending loop_body
      | _ -> ()) body in
  shared (Hashtbl.create 8) (Hashtbl.create 8) k.body;
  if !errors = [] then
    List.iter (fun (code, message) -> errors := { code; message } :: !errors)
      (Uniformity.check k);
  if !errors = [] then Ok () else Error (List.rev !errors)

let check_call = Call_verifier.check_call

let verify_exn k = match verify_kernel k with Ok () -> () | Error es -> raise (Invalid_kernel es)

let () =
  Printexc.Safe.register_printer (function
    | Invalid_kernel errors ->
        Some
          (String.concat "\n"
             (List.map (fun e -> e.code ^ ": " ^ e.message) errors))
    | _ -> None)
