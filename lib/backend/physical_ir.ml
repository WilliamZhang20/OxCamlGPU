(* Physicalization expands one logical tensor SSA value into the scalar values
   held by each lane. Supports 1D and rank-2 f32 tiles on one 32-lane warp. *)
type register = { index : int; dtype : Gpu_type.dtype option; value : Ir.value }
type value = {
  semantic : Ir.value;
  registers : register list;
  layout_mapping : Layout.mapped_register option;
}
type kernel = { target : Ptx_ir.kernel; values : (int * value) list }

type tile_slot_address =
  | Linear of { base : int; lane_stride : int }
  | Rank2 of {
      tile_cols : int;
      row_offset : int;
      col_offset : int;
      row_stride : int;
    }

(* A plain register distribution is the Blocked mapping of the same three
   factor lists, so normalizing here lets everything downstream read mapping
   fields directly instead of a per-field accessor per layout constructor. *)
let mapping_of_register_layout = function
  | Layout.Mapped_register mapping -> Some mapping
  | Layout.Register distribution -> Some {
      Layout.elements_per_lane=distribution.elements_per_lane;
      lanes_per_subgroup=distribution.lanes_per_subgroup;
      subgroups_per_cta=distribution.subgroups_per_cta;
      lane_mapping=Layout.Blocked;
      lane_order=distribution.order;
      register_order=distribution.order;
      subgroup_order=distribution.order }
  | Layout.Shared _ | Layout.Padded_shared _ | Layout.Global _ -> None

let mapping_for_layout layout = Option.bind layout mapping_of_register_layout

let max_value_id target =
  List.fold_left (fun n v -> max n v.Ir.id) (-1) (Ptx_ir.all_values target)

let register_mapping what layout =
  match mapping_of_register_layout layout with
  | Some mapping -> mapping
  | None ->
      invalid_arg
        (Printf.sprintf "%s requires a register layout, got %s" what
           (Layout.string_of_t layout))

(* Registers per lane of a mapping. Physicalization models exactly one
   32-lane subgroup per CTA, so anything else is refused here. *)
let one_warp_register_count mapping =
  let product = List.fold_left ( * ) 1 in
  if product mapping.Layout.lanes_per_subgroup <> 32
     || product mapping.Layout.subgroups_per_cta <> 1 then
    invalid_arg "physicalization currently requires one 32-lane subgroup per CTA";
  product mapping.Layout.elements_per_lane

let tensor_register_count value = match value.Ir.ty, value.Ir.layout with
  | Gpu_type.Tensor (shape, _), Some layout ->
      let elements =
        one_warp_register_count (register_mapping "physicalization" layout) in
      ignore (Layout.static_extents shape);
      elements
  | Gpu_type.Tensor _, _ -> invalid_arg "physicalization requires a tensor with a register layout"
  | _ -> 1

let iter_coordinates extents f =
  let rank = List.length extents in
  let rec loop coords dim =
    if dim=rank then f (List.rev coords)
    else
      let extent=List.nth extents dim in
      for i=0 to extent-1 do loop (i::coords) (dim+1) done
  in
  loop [] 0

(* Per-(lane, register) logical offsets of a mapped tile: slot
   [lane * elements + register] holds the memory offset of the element that
   lane keeps in that register. Every slot must be claimed at most once, and
   only by a lane of the one warp (CTA coordinates all zero). *)
let lane_register_offsets mapping ~elements extents strides =
  let offsets = Array.make (32 * elements) (-1) in
  iter_coordinates extents (fun logical ->
    let hardware = Layout.mapped_to_hardware mapping logical in
    if List.exists (fun c -> c <> 0) hardware.cta || hardware.subgroup_id <> 0 ||
       hardware.lane_id < 0 || hardware.lane_id >= 32 ||
       hardware.register_id < 0 || hardware.register_id >= elements then
      invalid_arg "physicalization received an unsupported tile mapping";
    let slot = hardware.lane_id * elements + hardware.register_id in
    if offsets.(slot) <> -1 then
      invalid_arg "physicalization layout maps multiple elements to one lane register";
    offsets.(slot) <- Layout.linear_offset strides logical);
  offsets

(* One [Linear] address per register when the offsets are affine in the lane,
   or [None] when some register's lanes do not sit on a line. *)
let affine_addresses offsets ~elements =
  let affine index =
    let base = offsets.(index) and next = offsets.(elements + index) in
    let lane_stride = next - base in
    let on_line lane = offsets.(lane * elements + index) = base + lane * lane_stride in
    if List.for_all on_line (List.init 32 Fun.id) then
      Some (Linear { base; lane_stride })
    else None
  in
  let addresses = List.init elements affine in
  if List.for_all Option.is_some addresses then
    Some (List.filter_map Fun.id addresses)
  else None

(* A tile whose lane [r * cols + c] owns element (r, c) is read with plain
   row-major addressing, no per-register table needed. *)
let row_major_address mapping ~elements extents ~cols ~row_stride =
  let lane_matches logical =
    match logical with
    | [r; c] -> (Layout.mapped_to_hardware mapping logical).lane_id = r * cols + c
    | _ -> false
  in
  let all_match = ref true in
  iter_coordinates extents (fun logical ->
    if not (lane_matches logical) then all_match := false);
  if elements = 1 && !all_match then
    Some (Rank2 { tile_cols = cols; row_offset = 0; col_offset = 0; row_stride })
  else None

let tile_slot_addresses mapping extents strides =
  let elements = one_warp_register_count mapping in
  let offsets () = lane_register_offsets mapping ~elements extents strides in
  match extents, strides with
  | [_], [_] ->
      (match affine_addresses (offsets ()) ~elements with
       | Some addresses -> addresses
       | None ->
           invalid_arg
             "physicalization: layout is not affine in lane and register coordinates")
  | [_; cols], [row_stride; col_stride] ->
      if col_stride <> 1 then
        invalid_arg "physicalization currently requires unit stride on the fastest MemRef dimension";
      let offsets = offsets () in
      (match row_major_address mapping ~elements extents ~cols ~row_stride with
       | Some address -> [address]
       | None ->
           match affine_addresses offsets ~elements with
           | Some addresses -> addresses
           | None ->
               invalid_arg "rank-2 physicalization requires row-major lane ownership or affine linear addressing")
  | _ -> invalid_arg "physicalization currently supports rank-1 or rank-2 tiles only"

let scalar_dtype value = match value.Ir.ty with
  | Gpu_type.F32 -> Some Gpu_type.Float32
  | Gpu_type.I32 | Gpu_type.Bool -> Some Gpu_type.Int32
  | _ -> None

(* The scalar register that stands for a tensor's [index]th per-lane element,
   or for the tensor itself when it is already a scalar. Slot 0 reuses the
   semantic value's id, so scalar code keeps its names. *)
let register_type = function
  | Gpu_type.Float32 -> Gpu_type.F32
  | Gpu_type.Int32 -> Gpu_type.I32
  (* Narrow floats have no register form: they are a storage format that TMA
     moves and WGMMA reads from shared memory. *)
  | Gpu_type.Bfloat16 | Gpu_type.Float16 ->
      invalid_arg "narrow float tiles have no register representation"

let value_registers ~next_id semantic =
  match semantic.Ir.ty with
  | Gpu_type.Tensor (_, dtype) ->
      List.init (tensor_register_count semantic) (fun index ->
        let ty = register_type dtype in
        let value =
          if index = 0 then { semantic with Ir.ty; layout = None }
          else begin
            let id = !next_id in
            incr next_id;
            Ir.make_value id ty
          end
        in
        { index; dtype = Some dtype; value })
  | _ -> [{ index = 0; dtype = scalar_dtype semantic; value = semantic }]

let physicalize target =
  let next_id = ref (max_value_id target + 1) in
  let table = Hashtbl.create 32 in
  List.iter
    (fun semantic ->
      if not (Hashtbl.mem table semantic.Ir.id) then
        Hashtbl.add table semantic.Ir.id
          { semantic;
            registers = value_registers ~next_id semantic;
            layout_mapping = mapping_for_layout semantic.Ir.layout })
    (Ptx_ir.all_values target);
  (* Every value reachable from the target was registered above, so a miss is
     a bug in this pass. Say which value, the way the slot lookup below does,
     rather than letting Not_found escape the compiler. *)
  let physical id =
    match Hashtbl.find_opt table id with
    | Some entry -> entry
    | None ->
        invalid_arg
          (Printf.sprintf "physicalization: value %d was never registered" id) in
  let register_count tensor = List.length (physical tensor.Ir.id).registers in
  let slot value index =
    match List.find_opt (fun register -> register.index=index) (physical value.Ir.id).registers with
    | Some register -> register.value
    | None -> invalid_arg (Printf.sprintf "physicalization: tensor %d has no register slot %d" value.Ir.id index) in
  let scalar value =
    match (physical value.Ir.id).registers with
    | [register] -> register.value
    | _ -> invalid_arg (Printf.sprintf "physicalization: tensor %d used as a scalar" value.Ir.id) in
  let addresses tensor memref =
    let mapping=match (physical tensor.Ir.id).layout_mapping with
      | Some mapping -> mapping
      | None -> invalid_arg "physicalization: tensor has no layout mapping" in
    let shape=match tensor.Ir.ty with
      | Gpu_type.Tensor (shape, _) -> shape
      | _ -> invalid_arg "tile addressing requires a tensor" in
    let strides=match memref.Ir.ty with
      | Gpu_type.MemRef (mem_shape, _, _) ->
          if mem_shape <> shape then
            invalid_arg "tile MemRef shape must match the tensor shape";
          Layout.memref_strides mem_shape memref.Ir.layout
      | _ -> invalid_arg "tile addressing requires a MemRef" in
    tile_slot_addresses mapping (Layout.static_extents shape) strides in
  (* One operation per register of [tensor], addressed by [memref]. *)
  let per_tile_slot tensor memref ~linear ~rank2 =
    List.mapi
      (fun index -> function
        | Linear { base; lane_stride } -> linear index base lane_stride
        | Rank2 { tile_cols; row_offset; col_offset; row_stride } ->
            rank2 index tile_cols row_offset col_offset row_stride)
      (addresses tensor memref)
  in
  let masked_needs_rank2 what _ _ _ =
    invalid_arg ("masked tile " ^ what ^ " currently require rank-2 tiles") in
  let lower_operation = function
    | Ptx_ir.Tile_load_f32 (tensor, memref) ->
        per_tile_slot tensor memref
          ~linear:(fun i base lane_stride ->
            Ptx_ir.Tile_load_f32_indexed
              (slot tensor i, scalar memref, base, lane_stride))
          ~rank2:(fun i cols ro co rs ->
            Ptx_ir.Tile_load_f32_2d
              (slot tensor i, scalar memref, cols, ro, co, rs, None))
    | Ptx_ir.Tile_store_f32 (memref, tensor) ->
        per_tile_slot tensor memref
          ~linear:(fun i base lane_stride ->
            Ptx_ir.Tile_store_f32_indexed
              (scalar memref, slot tensor i, base, lane_stride))
          ~rank2:(fun i cols ro co rs ->
            Ptx_ir.Tile_store_f32_2d
              (scalar memref, slot tensor i, cols, ro, co, rs, None))
    | Ptx_ir.Tile_load_f32_masked (tensor, memref, rows, cols_bound) ->
        per_tile_slot tensor memref
          ~linear:(masked_needs_rank2 "loads")
          ~rank2:(fun i cols ro co rs ->
            Ptx_ir.Tile_load_f32_2d
              ( slot tensor i, scalar memref, cols, ro, co, rs
              , Some (scalar rows, scalar cols_bound) ))
    | Ptx_ir.Tile_store_f32_masked (memref, tensor, rows, cols_bound) ->
        per_tile_slot tensor memref
          ~linear:(masked_needs_rank2 "stores")
          ~rank2:(fun i cols ro co rs ->
            Ptx_ir.Tile_store_f32_2d
              ( scalar memref, slot tensor i, cols, ro, co, rs
              , Some (scalar rows, scalar cols_bound) ))
    | Ptx_ir.Tensor_scale_f32 (dst, src, factor) ->
        List.init (register_count dst) (fun i ->
          Ptx_ir.Mul_f32 (slot dst i, slot src i, scalar factor))
    | Ptx_ir.Tensor_mul_f32 (dst, left, right) ->
        List.init (register_count dst) (fun i ->
          Ptx_ir.Mul_f32 (slot dst i, slot left i, slot right i))
    | Ptx_ir.Tensor_lane_f32 (dst, tensor) ->
        (match List.init (register_count tensor) (slot tensor) with
         | [] -> invalid_arg "physicalization: tensor has no registers"
         | first :: rest ->
             Ptx_ir.Move (scalar dst, first)
             :: List.map
                  (fun register ->
                    Ptx_ir.Add_f32 (scalar dst, scalar dst, register))
                  rest)
    (* Everything else is scalar already: swap each value for its register. *)
    | op -> [Ptx_ir.map_values scalar op]
  in
  let body = List.concat_map lower_operation target.Ptx_ir.body in
  { target = { target with Ptx_ir.body = body };
    values = Hashtbl.fold (fun id value acc -> (id, value) :: acc) table [] }

let lookup kernel value = List.assoc_opt value.Ir.id kernel.values

let coordinate kernel value logical = match lookup kernel value with
  | Some { layout_mapping=Some mapping; _ } -> Layout.mapped_to_hardware mapping logical
  | Some _ -> invalid_arg "physical value has no tensor coordinate mapping"
  | None -> invalid_arg "value has no physical representation"
