(* The table of GPU primitives, assembled from the per-family modules. Every
   name [Typedtree_support] can resolve a call to must have a handler, so a
   primitive added to the recognized list but not implemented fails when the
   exporter starts rather than when a kernel first uses it. *)
let table : (string, Export_state.handler) Hashtbl.t =
  let table = Hashtbl.create 128 in
  List.iter
    (fun (name, handler) ->
      if Hashtbl.mem table name then
        invalid_arg ("primitive registered twice: " ^ name);
      Hashtbl.add table name handler)
    (Export_prims_memory.prims
     @ Export_prims_hopper.prims
     @ Export_prims_index.prims
     @ Export_prims_float.prims
     @ Export_arith.prims
     @ Export_control.prims);
  List.iter
    (fun (_, name) ->
      if not (Hashtbl.mem table name) then
        invalid_arg ("primitive recognized but not implemented: " ^ name))
    Typedtree_support.operations;
  table

let find name = Hashtbl.find_opt table name
