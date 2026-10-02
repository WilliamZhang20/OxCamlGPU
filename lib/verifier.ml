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
  let add code message = errors := { code; message=Source_span.to_string !current_loc ^ ": " ^ message } :: !errors in
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
    current_loc := a.value.loc;
    check_layout a.value;
    if a.value.id<0 then add "E_DUPLICATE_ARG_ID" "argument ids must be nonnegative";
    match a.value.ty with
    | Ptr (F32, Global) | F32 | I32 | Bool -> ()
    | ty when is_global_f32_memref ty -> ()
    | _ -> add "E_ARG_TYPE" ("argument " ^ a.name ^ " has an unsupported GPU kernel ABI type")) k.args;
  let ids = List.sort compare arg_ids in
  let rec duplicate = function x :: y :: _ when x = y -> true | _ :: tl -> duplicate tl | [] -> false in
  if duplicate ids then add "E_DUPLICATE_ARG_ID" "argument value ids must be distinct";
  let seen = Hashtbl.create 16 in
  let gep_indices = Hashtbl.create 16 in
  List.iter (fun a -> Hashtbl.replace seen a.value.id a.value) k.args;
  let all_ids=Hashtbl.copy seen in
  let check_use seen v =
    match Hashtbl.find_opt seen v.id with
    | None -> add "E_UNDEFINED" (Printf.sprintf "value %d is used before definition or outside its scope" v.id)
    | Some canonical when canonical <> v -> add "E_VALUE_IDENTITY" "operand disagrees with canonical SSA definition"
    | Some _ -> () in
  let rec visit seen nested body =
  List.iter (fun i ->
    current_loc := (match results i @ operands i with v::_->v.loc | []->Source_span.synthetic);
    List.iter check_layout (operands i @ results i);
    List.iter (check_use seen) (operands i);
    (match i with
    | Const_bool(dst,_) -> if dst.ty <> Bool then add "E_CONST_TYPE" "bool constant must define bool"
    | Compare(dst,cmp,a,b) ->
        if dst.ty<>Bool || a.ty<>b.ty || not (List.mem a.ty [I32;F32;Bool]) ||
           (a.ty=Bool && not (List.mem cmp [Eq;Ne])) then add "E_COMPARE_TYPE" "comparison operand/result mismatch"
    | If(dst,c,a,b) ->
        if c.ty<>Bool then add "E_CONDITION_TYPE" "if condition must be bool";
        (match dst with Some d when not (List.mem d.ty [I32;F32;Bool]) -> add "E_YIELD_TYPE" "branch results must be scalar" | _->());
        let branch_loc = !current_loc in
        let branch (r : Ir.region) =
          let scope=Hashtbl.copy seen in visit scope true r.body;
          current_loc := branch_loc;
          Option.iter (check_use scope) r.yield;
          match dst,r.yield with
          | None,None -> ()
          | Some d,Some v when d.ty=v.ty ->
              if d.ownership<>Aliased || d.provenance<>None ||
                 not (locality_satisfies ~required:d.locality v.locality) ||
                 not (domain_portability_satisfies ~required:d.domain_portability v.domain_portability) ||
                 not (gpu_boundary_satisfies ~required:d.gpu_boundary v.gpu_boundary) ||
                 not (satisfies ~required:d.permission v.permission) then
                add "E_YIELD_FACTS" "branch result strengthens yielded modal facts"
          | _ -> add "E_YIELD_TYPE" "branch yield must match declared if result" in
        branch a; branch b
    | Const_i32 (dst, n) ->
        if Int64.of_int n < -2147483648L || Int64.of_int n > 2147483647L then add "E_CONST_RANGE" "i32 literal is out of range";
        if dst.ty <> I32 then add "E_CONST_TYPE" "i32 constant must define an i32 value"
    | Const_f32 (dst, _) ->
        if dst.ty <> F32 then add "E_CONST_TYPE" "f32 constant must define an f32 value"
    | Thread_idx_x dst | Global_idx_x dst ->
        if dst.ty <> I32 then add "E_INDEX_TYPE" "thread and global indices must define i32 values"
    | Add_i32 (dst, a, b) | Sub_i32(dst,a,b) | Mul_i32(dst,a,b) ->
        if dst.ty <> I32 || a.ty <> I32 || b.ty <> I32 then
          add "E_ARITH_TYPE" "i32 add requires an i32 result and i32 operands"
    | Add_f32 (dst, a, b) ->
        if dst.ty <> F32 || a.ty <> F32 || b.ty <> F32 then
          add "E_ARITH_TYPE" "f32 add requires an f32 result and f32 operands"
    | Mul_f32 (dst, a, b) ->
        if dst.ty <> F32 || a.ty <> F32 || b.ty <> F32 then
          add "E_ARITH_TYPE" "f32 multiply requires an f32 result and f32 operands"
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
    | Gep_f32 (dst, ptr, ix) ->
        Hashtbl.replace gep_indices dst.id ix.id;
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
        if dst.ty <> src.ty ||
           (match dst.ty with Tensor (_, Float32) -> false | _ -> true) ||
           scalar.ty <> F32 then
          add "E_TILE_ARITH_TYPE" "f32 tile scaling requires matching Tensor values and an f32 scalar";
        (match dst.layout, src.layout with
         | Some (Layout.Register _), Some (Layout.Register _)
           when dst.layout = src.layout -> ()
         | _ -> add "E_TENSOR_LAYOUT" "tile scaling requires identical register layouts on input and output")
    | Mul_tensor_f32 (dst, a, b) ->
        if dst.ty <> a.ty || dst.ty <> b.ty ||
           (match dst.ty with Tensor (_, Float32) -> false | _ -> true) then
          add "E_TILE_ARITH_TYPE" "f32 tensor multiplication requires matching f32 Tensor values";
        (match dst.layout, a.layout, b.layout with
         | Some (Layout.Register _), Some (Layout.Register _), Some (Layout.Register _)
           when dst.layout = a.layout && a.layout = b.layout -> ()
         | _ -> add "E_TENSOR_LAYOUT" "tensor multiplication requires identical register layouts")
    | Reduce_sum_f32 (dst, src) ->
        (match dst.ty, src.ty with
         | F32, Tensor (_, Float32) -> ()
         | _ -> add "E_REDUCE_TYPE" "f32 sum requires an f32 Tensor and returns f32");
        (match src.layout with
         | Some (Layout.Register _) -> ()
         | _ -> add "E_TENSOR_LAYOUT" "reduction input requires a register layout")
    | Warp_sum_f32 (dst, src) ->
        if dst.ty <> F32 || src.ty <> F32 then
          add "E_WARP_REDUCE_TYPE" "warp sum currently reduces f32 values and returns f32"
    | Barrier _ -> ()
    | Return None when nested -> add "E_REGION_RETURN" "branch regions must yield, not return"
    | Return (Some _) -> add "E_KERNEL_RETURN_TYPE" "GPU kernel entry points must return unit"
    | _ -> ());
    List.iter (fun v ->
      if v.id<0 || Hashtbl.mem all_ids v.id then add "E_REDEFINITION" "invalid or duplicate SSA definition"
      else (Hashtbl.add all_ids v.id v; Hashtbl.add seen v.id v)) (results i)) body in
  visit seen false k.body;
  (* Definite initialization intersects at joins; pending writes union. *)
  let rec shared initialized pending body =
    List.iter (function
      | Shared_alloc m -> Hashtbl.replace initialized m.id false
      | Store_tensor(m,_) when (match m.ty with MemRef(_,_,Shared)->true|_->false) ->
          Hashtbl.replace initialized m.id true; Hashtbl.replace pending m.id true
      | Load_tensor(_,m) when (match m.ty with MemRef(_,_,Shared)->true|_->false) ->
          if Hashtbl.find_opt initialized m.id <> Some true then add "E_SHARED_UNINITIALIZED" "shared tile is not definitely initialized"
          else if Hashtbl.mem pending m.id then add "E_SHARED_SYNC" "shared tile is read before a CTA barrier"
      | Barrier Execution.Cta -> Hashtbl.clear pending
      | If(_,_,a,b) ->
          let ai=Hashtbl.copy initialized and ap=Hashtbl.copy pending in
          let bi=Hashtbl.copy initialized and bp=Hashtbl.copy pending in
          shared ai ap a.body; shared bi bp b.body;
          Hashtbl.iter (fun id _ -> Hashtbl.replace initialized id
            (Hashtbl.find_opt ai id=Some true && Hashtbl.find_opt bi id=Some true)) (Hashtbl.copy initialized);
          Hashtbl.clear pending;
          Hashtbl.iter (fun id _ -> Hashtbl.replace pending id true) ap;
          Hashtbl.iter (fun id _ -> Hashtbl.replace pending id true) bp
      | _ -> ()) body in
  shared (Hashtbl.create 8) (Hashtbl.create 8) k.body;
  if !errors=[] then List.iter (fun (code,message)->errors := {code;message} :: !errors) (Uniformity.check k);
  if !errors = [] then Ok () else Error (List.rev !errors)

let check_call = Call_verifier.check_call

let verify_exn k = match verify_kernel k with Ok () -> () | Error es -> raise (Invalid_kernel es)

let () = Printexc.register_printer (function
  | Invalid_kernel errors -> Some (String.concat "\n"
      (List.map (fun e -> e.code ^ ": " ^ e.message) errors))
  | _ -> None)
