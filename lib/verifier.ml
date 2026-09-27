open Gpu_type
open Gpu_ir

type error = { code : string; message : string }
exception Invalid_kernel of error list

let verify_kernel k =
  let errors = ref [] in
  let add code message = errors := { code; message } :: !errors in
  let arg_ids = List.map (fun a -> a.value.id) k.args in
  List.iter (fun a ->
    match a.value.ty with
    | Ptr (F32, Global) | F32 -> ()
    | _ -> add "E_ARG_TYPE" ("argument " ^ a.name ^ " must be a global f32 pointer or f32 scalar")) k.args;
  let ids = List.sort compare arg_ids in
  let rec duplicate = function x :: y :: _ when x = y -> true | _ :: tl -> duplicate tl | [] -> false in
  if duplicate ids then add "E_DUPLICATE_ARG_ID" "argument value ids must be distinct";
  let seen = Hashtbl.create 16 in
  List.iter (fun a -> Hashtbl.replace seen a.value.id a.value) k.args;
  List.iter (fun i ->
    List.iter (fun v ->
      if not (Hashtbl.mem seen v.id) then add "E_UNDEFINED" (Printf.sprintf "value %d is used before definition" v.id)) (operands i);
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
    | Load_f32 (_, ptr) ->
        (match ptr.ty with Ptr (F32, Global) -> () | _ -> add "E_LOAD_TYPE" "load requires a global f32 pointer");
        if ptr.permission = Write_only || ptr.permission = Immutable then
          add "E_UNREADABLE_LOAD" (Printf.sprintf "load through non-readable value %d" ptr.id)
    | Gep_f32 (dst, ptr, ix) ->
        if dst.ty <> Ptr (F32, Global) || ptr.ty <> Ptr (F32, Global) || ix.ty <> I32 then
          add "E_GEP_TYPE" "f32 element address requires global f32 pointers and i32 index";
        if dst.permission <> ptr.permission || dst.ownership <> ptr.ownership ||
           dst.domain_portability <> ptr.domain_portability || dst.gpu_boundary <> ptr.gpu_boundary ||
           (ptr.locality = Local && dst.locality <> Local) then
          add "E_GEP_FACTS" "element address must preserve its base ownership/access facts and cannot widen locality"
    | Add_i32 (dst, a, b) -> if dst.ty <> I32 || a.ty <> I32 || b.ty <> I32 then add "E_ARITH_TYPE" "i32 add requires i32 values"
    | Add_f32 (dst, a, b) | Mul_f32 (dst, a, b) -> if dst.ty <> F32 || a.ty <> F32 || b.ty <> F32 then add "E_ARITH_TYPE" "f32 arithmetic requires f32 values"
    | Return (Some v) when v.locality = Local -> add "E_LOCAL_ESCAPE" (Printf.sprintf "local value %d escapes kernel scope" v.id)
    | _ -> ()) k.body;
  if !errors = [] then Ok () else Error (List.rev !errors)

let check_call k actuals =
  let errors = ref [] in
  if List.length actuals <> List.length k.args then
    errors := [{ code="E_ARITY"; message="kernel call has the wrong number of arguments" }]
  else List.iter2 (fun formal actual ->
    if formal.value.ownership = Unique && actual.actual_ownership <> Unique then
      errors := { code="E_UNIQUE_REQUIRED"; message="argument " ^ formal.name ^ " requires unique access" } :: !errors;
    let can_read = function Read_only | Read_write -> true | Write_only | Immutable -> false in
    let can_write = function Write_only | Read_write -> true | Read_only | Immutable -> false in
    let permitted = match formal.value.permission with
      | Read_only -> can_read actual.actual_permission
      | Write_only -> can_write actual.actual_permission
      | Read_write -> can_read actual.actual_permission && can_write actual.actual_permission
      | Immutable -> actual.actual_permission = Immutable in
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
