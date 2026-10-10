open Gpu_mode
open Ir
open Verification_error

let check_call kernel actuals =
  let errors = ref [] in
  let add code message = errors := { code; message } :: !errors in
  let require_fact code description ok (formal : arg) =
    if not ok then
      add code ("argument " ^ formal.name ^ " requires " ^ description)
  in
  if List.length actuals <> List.length kernel.args then
    add "E_ARITY" "kernel call has the wrong number of arguments"
  else begin
    List.iter2 (fun formal actual ->
      if formal.value.ownership = Unique && actual.actual_ownership <> Unique then
        add "E_UNIQUE_REQUIRED"
          ("argument " ^ formal.name ^ " requires unique access");
      if not (satisfies ~required:formal.value.permission actual.actual_permission) then
        add "E_ACCESS_REQUIRED"
          ("argument " ^ formal.name ^ " requires " ^
           (match formal.value.permission with
            | Read_only -> "read"
            | Write_only -> "write"
            | Read_write -> "read_write"
            | Immutable -> "immutable") ^ " access");
      require_fact "E_LOCALITY_REQUIRED"
        (match formal.value.locality with
         | Local -> "local lifetime"
         | Global -> "global lifetime")
        (locality_satisfies ~required:formal.value.locality actual.actual_locality)
        formal;
      require_fact "E_DOMAIN_PORTABILITY_REQUIRED"
        "a compatible OxCaml domain-portability mode"
        (domain_portability_satisfies
           ~required:formal.value.domain_portability
           actual.actual_domain_portability)
        formal;
      require_fact "E_GPU_BOUNDARY_REQUIRED"
        "a compatible GPU-boundary mode"
        (gpu_boundary_satisfies
           ~required:formal.value.gpu_boundary actual.actual_gpu_boundary)
        formal)
      kernel.args actuals;

    let unique_roots = List.filter_map (fun (formal, actual) ->
      if formal.value.ownership = Unique then Some actual.buffer_id else None)
      (List.combine kernel.args actuals)
    in
    let instantiate formal actual =
      { formal.value with
        provenance = Some actual.buffer_id;
        ownership = actual.actual_ownership }
    in
    let rec check_unique_pairs formals actuals =
      match formals, actuals with
      | formal :: rest_formals, actual :: rest_actuals ->
          let rec check_later later_formals later_actuals =
            match later_formals, later_actuals with
            | later_formal :: more_formals, later_actual :: more_actuals ->
                if (formal.value.ownership = Unique ||
                    later_formal.value.ownership = Unique) &&
                   Alias_analysis.may_alias ~unique_roots kernel
                     (instantiate formal actual)
                     (instantiate later_formal later_actual)
                then
                  add "E_UNIQUE_ALIAS"
                    (Printf.sprintf
                       "buffer %d is passed to distinct unique arguments %s and %s"
                       actual.buffer_id formal.name later_formal.name);
                check_later more_formals more_actuals
            | _ -> ()
          in
          check_later rest_formals rest_actuals;
          check_unique_pairs rest_formals rest_actuals
      | _ -> ()
    in
    check_unique_pairs kernel.args actuals
  end;
  if !errors = [] then Ok () else Error (List.rev !errors)
