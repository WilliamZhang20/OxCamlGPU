open Gpu_type
open Mode
open Ir

type error = { code : string; message : string }
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

let may_alias k a b =
  match a.provenance, b.provenance with
  | Some a_root, Some b_root when a_root = b_root -> true
  | Some a_root, Some b_root ->
      let is_unique_root root =
        List.exists (fun arg -> arg.value.id = root && arg.value.ownership = Unique) k.args in
      not (is_unique_root a_root || is_unique_root b_root)
  | _ -> true

let verify_kernel k =
  let errors = ref [] in
  let add code message = errors := { code; message } :: !errors in
  let check_layout value =
    let validate expected shape layout =
      (match layout with
       | layout when expected layout ->
           (match Layout.validate shape layout with
            | Ok () -> ()
            | Error message -> add "E_LAYOUT_INVALID" (Printf.sprintf "value %d: %s" value.id message))
       | _ -> add "E_LAYOUT_KIND" (Printf.sprintf "value %d has a layout incompatible with its type" value.id)) in
    match value.ty, value.layout with
    | Tensor (shape, _), Some ((Layout.Register _) as layout) ->
        validate (function Layout.Register _ -> true | _ -> false) shape layout
    | Tensor _, _ -> add "E_TENSOR_LAYOUT" (Printf.sprintf "tensor value %d requires a register layout" value.id)
    | MemRef (shape, _, Shared), Some ((Layout.Shared _) as layout) ->
        validate (function Layout.Shared _ -> true | _ -> false) shape layout
    | MemRef (_, _, Shared), _ -> add "E_SHARED_LAYOUT" (Printf.sprintf "shared MemRef %d requires a shared layout" value.id)
    | MemRef (_, _, (Global | Local)), Some _ ->
        add "E_MEMREF_LAYOUT" "explicit MemRef layouts are currently defined only for shared storage"
    | _, Some _ -> add "E_LAYOUT_KIND" (Printf.sprintf "scalar/pointer value %d cannot carry a tile layout" value.id)
    | _, None -> () in
  let arg_ids = List.map (fun a -> a.value.id) k.args in
  List.iter (fun a ->
    check_layout a.value;
    match a.value.ty with
    | Ptr (F32, Global) | F32 -> ()
    | ty when is_global_f32_memref ty -> ()
    | _ -> add "E_ARG_TYPE" ("argument " ^ a.name ^ " must be a global f32 pointer or f32 scalar")) k.args;
  let ids = List.sort compare arg_ids in
  let rec duplicate = function x :: y :: _ when x = y -> true | _ :: tl -> duplicate tl | [] -> false in
  if duplicate ids then add "E_DUPLICATE_ARG_ID" "argument value ids must be distinct";
  let seen = Hashtbl.create 16 in
  List.iter (fun a -> Hashtbl.replace seen a.value.id a.value) k.args;
  List.iter (fun i ->
    List.iter check_layout (operands i @ results i);
    List.iter (fun v ->
      match Hashtbl.find_opt seen v.id with
      | None -> add "E_UNDEFINED" (Printf.sprintf "value %d is used before definition" v.id)
      | Some canonical when canonical <> v ->
          add "E_VALUE_IDENTITY" (Printf.sprintf "operand %d disagrees with its canonical SSA value" v.id)
      | Some _ -> ()) (operands i);
    List.iter (fun v ->
      if Hashtbl.mem seen v.id then add "E_REDEFINITION" (Printf.sprintf "value %d is defined more than once" v.id)
      else Hashtbl.add seen v.id v) (results i);
    match i with
    | Store_f32 (ptr, value) ->
        (match ptr.ty, value.ty with
         | Ptr (F32, Global), F32 -> ()
         | _ -> add "E_STORE_TYPE" "store requires a global f32 pointer and f32 value");
        if ptr.permission = Read_only || ptr.permission = Immutable then
          add "E_READONLY_STORE" (Printf.sprintf "store through non-writable value %d" ptr.id)
    | Store_f32_lane0 (ptr, value) ->
        (match ptr.ty, value.ty with
         | (Ptr (F32, Global) | MemRef (_, Float32, Global)), F32 -> ()
         | _ -> add "E_STORE_TYPE" "lane-zero store requires a global f32 buffer and f32 value");
        if not (writable ptr.permission) then
          add "E_READONLY_STORE" (Printf.sprintf "lane-zero store through non-writable value %d" ptr.id)
    | Load_f32 (_, ptr) ->
        (match ptr.ty with Ptr (F32, Global) -> () | _ -> add "E_LOAD_TYPE" "load requires a global f32 pointer");
        if ptr.permission = Write_only then
          add "E_UNREADABLE_LOAD" (Printf.sprintf "load through non-readable value %d" ptr.id)
    | Gep_f32 (dst, ptr, ix) ->
        if dst.ty <> Ptr (F32, Global) || not (is_global_f32_buffer ptr.ty) || ix.ty <> I32 then
          add "E_GEP_TYPE" "f32 element address requires global f32 pointers and i32 index";
        if dst.permission <> ptr.permission || dst.ownership <> Aliased ||
           dst.provenance <> ptr.provenance ||
           dst.domain_portability <> ptr.domain_portability || dst.gpu_boundary <> ptr.gpu_boundary ||
           (ptr.locality = Local && dst.locality <> Local) then
          add "E_GEP_FACTS" "element address must be aliased, retain root provenance/access facts, and cannot widen locality"
    | Shared_alloc memref ->
        (match memref_type memref.ty with
         | Some (shape, _, Shared) ->
             if not (is_static_shape shape) then
               add "E_SHARED_DYNAMIC_ALLOC" "shared allocation requires a fully static shape"
         | _ -> add "E_SHARED_ALLOC_TYPE" "shared allocation must define a shared-space MemRef");
        (match memref.permission with
         | Read_only | Immutable -> add "E_SHARED_ALLOC_PERMISSION" "shared allocation must be writable"
         | Write_only | Read_write -> ())
    | Load_tensor (dst, src) ->
        (match tile_type dst.ty, memref_type src.ty with
         | Some (dst_shape, dst_dtype), Some (src_shape, src_dtype, space) ->
             if dst_shape <> src_shape || dst_dtype <> src_dtype then
               add "E_TILE_TRANSFER_TYPE" "tile load source and destination must have matching shape and dtype";
             if not (readable src.permission) then
               add "E_UNREADABLE_LOAD" (Printf.sprintf "tile load through non-readable MemRef %d" src.id);
             (match space, src.layout with
              | Shared, Some (Layout.Shared _) -> ()
              | Shared, _ -> add "E_SHARED_LAYOUT" "shared tile source requires a shared layout"
              | (Global | Local), _ -> ())
         | _ -> add "E_TILE_LOAD_TYPE" "tile load requires a Tensor destination and MemRef source");
        (match dst.layout with Some (Layout.Register _) -> () | _ -> add "E_TENSOR_LAYOUT" "tile load destination requires a register layout")
    | Store_tensor (dst, src) ->
        (match memref_type dst.ty, tile_type src.ty with
         | Some (dst_shape, dst_dtype, space), Some (src_shape, src_dtype) ->
             if dst_shape <> src_shape || dst_dtype <> src_dtype then
               add "E_TILE_TRANSFER_TYPE" "tile store source and destination must have matching shape and dtype";
             if not (writable dst.permission) then
               add "E_READONLY_STORE" (Printf.sprintf "tile store through non-writable MemRef %d" dst.id);
             (match space, dst.layout with
              | Shared, Some (Layout.Shared _) -> ()
              | Shared, _ -> add "E_SHARED_LAYOUT" "shared tile destination requires a shared layout"
              | (Global | Local), _ -> ())
         | _ -> add "E_TILE_STORE_TYPE" "tile store requires a MemRef destination and Tensor source");
        (match src.layout with Some (Layout.Register _) -> () | _ -> add "E_TENSOR_LAYOUT" "tile store source requires a register layout")
    | Scale_tensor_f32 (dst, src, scalar) ->
        if dst.ty <> src.ty || tile_type dst.ty = None || scalar.ty <> F32 then
          add "E_TILE_ARITH_TYPE" "f32 tile scaling requires matching Tensor values and an f32 scalar";
        (match dst.layout, src.layout with
         | Some (Layout.Register _), Some (Layout.Register _) -> ()
         | _ -> add "E_TENSOR_LAYOUT" "tile scaling requires register layouts on input and output")
    | Warp_reduce_sum_f32 (dst, src) ->
        if dst.ty <> F32 || src.ty <> F32 then
          add "E_WARP_REDUCE_TYPE" "warp sum currently reduces f32 values and returns f32"
    | Barrier _ -> ()
    | Add_i32 (dst, a, b) -> if dst.ty <> I32 || a.ty <> I32 || b.ty <> I32 then add "E_ARITH_TYPE" "i32 add requires i32 values"
    | Add_f32 (dst, a, b) | Mul_f32 (dst, a, b) -> if dst.ty <> F32 || a.ty <> F32 || b.ty <> F32 then add "E_ARITH_TYPE" "f32 arithmetic requires f32 values"
    | Return (Some v) when v.locality = Local -> add "E_LOCAL_ESCAPE" (Printf.sprintf "local value %d escapes kernel scope" v.id)
    | _ -> ()) k.body;
  (* The current IR is straight-line. A CTA barrier is required between a
     shared-memory tile write and a later read, which is conservative even
     when both operations happen in the same warp. *)
  let shared_initialized = Hashtbl.create 8 and shared_pending = Hashtbl.create 8 in
  List.iter (function
    | Shared_alloc memref -> Hashtbl.replace shared_initialized memref.id false
    | Store_tensor (dst, _) when (match memref_type dst.ty with Some (_, _, Shared) -> true | _ -> false) ->
        Hashtbl.replace shared_initialized dst.id true;
        Hashtbl.replace shared_pending dst.id true
    | Load_tensor (_, src) when (match memref_type src.ty with Some (_, _, Shared) -> true | _ -> false) ->
        if not (Hashtbl.mem shared_initialized src.id && Hashtbl.find shared_initialized src.id) then
          add "E_SHARED_UNINITIALIZED" (Printf.sprintf "shared MemRef %d is read before a tile store" src.id)
        else if Hashtbl.mem shared_pending src.id then
          add "E_SHARED_SYNC" (Printf.sprintf "shared MemRef %d is read before a CTA barrier" src.id)
    | Barrier Cta -> Hashtbl.clear shared_pending
    | _ -> ()) k.body;
  if !errors = [] then Ok () else Error (List.rev !errors)

let check_call k actuals =
  let errors = ref [] in
  if List.length actuals <> List.length k.args then
    errors := [{ code="E_ARITY"; message="kernel call has the wrong number of arguments" }]
  else List.iter2 (fun formal actual ->
    if formal.value.ownership = Unique && actual.actual_ownership <> Unique then
      errors := { code="E_UNIQUE_REQUIRED"; message="argument " ^ formal.name ^ " requires unique access" } :: !errors;
    let permitted = satisfies ~required:formal.value.permission actual.actual_permission in
    if not permitted then
      errors := { code="E_ACCESS_REQUIRED"; message="argument " ^ formal.name ^ " requires " ^
        (match formal.value.permission with Read_only -> "read" | Write_only -> "write" | Read_write -> "read_write" | Immutable -> "immutable") ^ " access" } :: !errors) k.args actuals;
  let rec pairs formals actuals = match formals, actuals with
    | f::ft, a::at ->
        let rec rest fs as_ = match fs, as_ with
          | f2::fs2, a2::as2 ->
              if (f.value.ownership = Unique || f2.value.ownership = Unique) && a.buffer_id = a2.buffer_id then
                errors := { code="E_UNIQUE_ALIAS"; message=Printf.sprintf "buffer %d is passed to distinct unique arguments %s and %s" a.buffer_id f.name f2.name } :: !errors;
              rest fs2 as2
          | _ -> () in rest ft at; pairs ft at
    | _ -> () in
  if List.length actuals = List.length k.args then pairs k.args actuals;
  if !errors = [] then Ok () else Error (List.rev !errors)

let verify_exn k = match verify_kernel k with Ok () -> () | Error es -> raise (Invalid_kernel es)
