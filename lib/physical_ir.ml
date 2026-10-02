(* Physicalization records how each semantic SSA value occupies registers on
   one hardware participant. It is deliberately separate from semantic IR:
   tensor values map to register tuples, while scalar values map to one slot. *)
type register = { index : int; dtype : Gpu_type.dtype }
type value = { semantic : Ir.value; registers : register list }
type kernel = {
  target : Ptx_ir.kernel;
  values : (int * value) list;
}

let logical_elements shape = Gpu_type.shape_elements shape

let registers_for value = match value.Ir.ty, value.Ir.layout with
  | Gpu_type.Tensor (shape, dtype), Some (Layout.Register layout) ->
      let count = List.fold_left ( * ) 1 layout.Layout.elements_per_lane in
      if count <> 1 then
        invalid_arg "physicalization: PTX backend does not yet expand multi-register tensor values";
      let expected = logical_elements shape in
      let covered = List.fold_left ( * ) 1 layout.Layout.lanes_per_subgroup
        * List.fold_left ( * ) 1 layout.Layout.subgroups_per_cta in
      if count * covered <> expected then invalid_arg "physicalization: register layout does not cover tensor shape";
      List.init count (fun index -> { index; dtype })
  | Gpu_type.Tensor _, _ -> invalid_arg "physicalization: tensor has no register layout"
  | Gpu_type.F32, _ -> [{ index=0; dtype=Gpu_type.Float32 }]
  | (Gpu_type.I32 | Gpu_type.Bool), _ -> [{ index=0; dtype=Gpu_type.Int32 }]
  | _ -> []

let physicalize target =
  let table = Hashtbl.create 32 in
  let add value =
    let physical = { semantic=value; registers=registers_for value } in
    Hashtbl.replace table value.Ir.id physical in
  List.iter (fun arg -> add arg.Ir.value) target.Ptx_ir.args;
  List.iter (fun operation ->
    List.iter add (Ptx_ir.results operation);
    List.iter add (Ptx_ir.operands operation)) target.Ptx_ir.body;
  { target; values=Hashtbl.fold (fun id value acc -> (id,value)::acc) table [] }

let lookup kernel value = List.assoc_opt value.Ir.id kernel.values
