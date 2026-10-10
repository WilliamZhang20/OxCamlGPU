type swizzle = No_swizzle | Swizzle_32B | Swizzle_64B | Swizzle_128B

type register = {
  (* A target-independent distribution: each logical dimension is split
     across per-lane elements, lanes in a subgroup, and subgroups in a CTA. *)
  elements_per_lane : int list;
  lanes_per_subgroup : int list;
  subgroups_per_cta : int list;
  order : int list;
}

type lane_mapping = Blocked | Interleaved

type mapped_register = {
  elements_per_lane : int list;
  lanes_per_subgroup : int list;
  subgroups_per_cta : int list;
  lane_mapping : lane_mapping;
  (* Each order lists dimensions from fastest-changing to slowest-changing. *)
  lane_order : int list;
  register_order : int list;
  subgroup_order : int list;
}

type shared = {
  vector_width : int;
  (* Dimension order for shared addresses, fastest-changing first. *)
  order : int list;
  swizzle : swizzle;
}

type padded_shared = { base : shared; row_padding_bytes : int }

(* Explicit element strides for a global MemRef. Rank must match the shape;
   ordinary row-major contiguous storage uses contiguous_strides. *)
type global = { strides : int list }

type t = Register of register | Mapped_register of mapped_register
  | Shared of shared | Padded_shared of padded_shared | Global of global

(* A logical coordinate is decomposed into coordinates at each hardware
   level. The per-dimension vectors keep this mapping compositional: target
   lowering may later linearize or swizzle each level for a specific GPU. *)
type hardware_coordinate = {
  cta : int list;
  subgroup : int list;
  lane : int list;
  register : int list;
  lane_id : int;
  register_id : int;
  subgroup_id : int;
}

let valid_order rank order =
  List.length order = rank
  && List.sort compare order = List.init rank Fun.id

let linearize order extents coordinates =
  List.fold_left (fun (stride, value) dimension ->
    stride * List.nth extents dimension,
    value + stride * List.nth coordinates dimension) (1, 0) order |> snd

let validate_mapping_distribution (distribution : register) =
  let rank = List.length distribution.elements_per_lane in
  if List.length distribution.lanes_per_subgroup <> rank ||
     List.length distribution.subgroups_per_cta <> rank ||
     not (List.for_all (fun n -> n > 0)
       (distribution.elements_per_lane @ distribution.lanes_per_subgroup @
        distribution.subgroups_per_cta)) then
    invalid_arg "register mapping factors must be positive and have matching ranks"

let validate_components extents coordinates what =
  if List.length extents <> List.length coordinates ||
     List.exists2 (fun n x -> x < 0 || x >= n) extents coordinates then
    invalid_arg (what ^ " coordinate is outside its layout extent")

let mapped_to_hardware mapping logical =
  let rank = List.length mapping.elements_per_lane in
  if List.length mapping.lanes_per_subgroup <> rank ||
     List.length mapping.subgroups_per_cta <> rank ||
     not (List.for_all (fun n -> n > 0)
       (mapping.elements_per_lane @ mapping.lanes_per_subgroup @ mapping.subgroups_per_cta)) then
    invalid_arg "mapped register factors must be positive and have matching ranks";
  if not (valid_order rank mapping.lane_order && valid_order rank mapping.register_order &&
          valid_order rank mapping.subgroup_order) then
    invalid_arg "mapped register axis orders must be permutations of tensor dimensions";
  if List.length logical <> rank then invalid_arg "logical coordinate rank does not match mapped layout";
  let lane, register, subgroup = List.fold_left (fun (lanes, registers, subgroups)
      (dimension, coordinate) ->
    let elements = List.nth mapping.elements_per_lane dimension
    and lane_extent=List.nth mapping.lanes_per_subgroup dimension
    and subgroup_extent=List.nth mapping.subgroups_per_cta dimension in
    if coordinate < 0 || coordinate >= elements * lane_extent * subgroup_extent then
      invalid_arg "logical coordinate is outside mapped layout extent";
    let subgroup, lane, register = match mapping.lane_mapping with
      | Blocked ->
          let within = coordinate mod (elements * lane_extent) in
          coordinate / (elements * lane_extent), within / elements, within mod elements
      | Interleaved ->
          let within = coordinate mod (elements * lane_extent) in
          coordinate / (elements * lane_extent), within mod lane_extent, within / lane_extent in
    let set xs value =
      let copy = Array.of_list xs in Array.set copy dimension value; Array.to_list copy in
    (set lanes lane, set registers register, set subgroups subgroup))
    (List.init rank (fun _ -> 0), List.init rank (fun _ -> 0), List.init rank (fun _ -> 0))
    (List.mapi (fun i x -> i,x) logical) in
  validate_components mapping.lanes_per_subgroup lane "lane";
  validate_components mapping.elements_per_lane register "register";
  validate_components mapping.subgroups_per_cta subgroup "subgroup";
  { cta = List.init rank (fun _ -> 0); lane; subgroup; register;
    lane_id = linearize mapping.lane_order mapping.lanes_per_subgroup lane;
    register_id = linearize mapping.register_order mapping.elements_per_lane register;
    subgroup_id = linearize mapping.subgroup_order mapping.subgroups_per_cta subgroup }

let logical_to_hardware (register : register) logical =
  validate_mapping_distribution register;
  let rank = List.length register.elements_per_lane in
  if not (valid_order rank register.order) then
    invalid_arg "register order must be a permutation of tensor dimensions";
  let factors = List.combine register.elements_per_lane
      (List.combine register.lanes_per_subgroup register.subgroups_per_cta) in
  if List.length logical <> List.length factors then
    invalid_arg "logical coordinate rank does not match register layout";
  let parts = List.map2 (fun coordinate (elements, (lanes, subgroups)) ->
    if coordinate < 0 || coordinate >= elements * lanes * subgroups then
      invalid_arg "logical coordinate is outside register layout extent";
    let per_lane, distributed =
      coordinate mod elements, coordinate / elements in
    let lane = distributed mod lanes and cta = distributed / lanes in
    let subgroup = cta mod subgroups and cta = cta / subgroups in
    per_lane, lane, subgroup, cta) logical factors in
  let register_values = List.map (fun (x,_,_,_) -> x) parts
  and lane=List.map (fun (_,x,_,_) -> x) parts
  and subgroup=List.map (fun (_,_,x,_) -> x) parts
  and cta=List.map (fun (_,_,_,x) -> x) parts in
  { register=register_values; lane; subgroup; cta;
    lane_id = linearize register.order register.lanes_per_subgroup lane;
    register_id = linearize register.order register.elements_per_lane register_values;
    subgroup_id = linearize register.order register.subgroups_per_cta subgroup }

let hardware_to_logical (layout : register) hardware =
  let rank = List.length layout.elements_per_lane in
  if List.exists (fun xs -> List.length xs <> rank)
      [hardware.cta;hardware.subgroup;hardware.lane;hardware.register] then
    invalid_arg "hardware coordinate rank does not match register layout";
  let logical = List.init rank (fun dimension ->
    let elements = List.nth layout.elements_per_lane dimension
    and lanes=List.nth layout.lanes_per_subgroup dimension
    and subgroups=List.nth layout.subgroups_per_cta dimension in
    let register=List.nth hardware.register dimension
    and lane=List.nth hardware.lane dimension
    and subgroup=List.nth hardware.subgroup dimension
    and cta=List.nth hardware.cta dimension in
    if register < 0 || register >= elements || lane < 0 || lane >= lanes ||
       subgroup < 0 || subgroup >= subgroups || cta <> 0 then
      invalid_arg "hardware coordinate is outside this CTA register layout";
    (((subgroup * lanes + lane) * elements) + register)) in
  let expected_lane = linearize layout.order layout.lanes_per_subgroup hardware.lane
  and expected_subgroup=linearize layout.order layout.subgroups_per_cta hardware.subgroup in
  let expected_register = linearize layout.order layout.elements_per_lane hardware.register in
  if hardware.lane_id <> expected_lane || hardware.subgroup_id <> expected_subgroup ||
     hardware.register_id <> expected_register then
    invalid_arg "linear hardware IDs disagree with per-dimension coordinates";
  logical

let mapped_hardware_to_logical mapping hardware =
  let rank = List.length mapping.elements_per_lane in
  if List.length mapping.lanes_per_subgroup <> rank ||
     List.length mapping.subgroups_per_cta <> rank ||
     not (List.for_all (fun n -> n > 0)
       (mapping.elements_per_lane @ mapping.lanes_per_subgroup @ mapping.subgroups_per_cta)) then
    invalid_arg "mapped register factors must be positive and have matching ranks";
  if not (valid_order rank mapping.lane_order && valid_order rank mapping.register_order &&
          valid_order rank mapping.subgroup_order) then
    invalid_arg "mapped register axis orders must be permutations of tensor dimensions";
  if List.exists (fun xs -> List.length xs <> rank)
      [hardware.cta;hardware.subgroup;hardware.lane;hardware.register] then
    invalid_arg "hardware coordinate rank does not match mapped layout";
  let expected_lane = linearize mapping.lane_order mapping.lanes_per_subgroup hardware.lane
  and expected_subgroup=linearize mapping.subgroup_order mapping.subgroups_per_cta hardware.subgroup in
  let expected_register = linearize mapping.register_order mapping.elements_per_lane hardware.register in
  if hardware.lane_id <> expected_lane || hardware.subgroup_id <> expected_subgroup ||
     hardware.register_id <> expected_register then
    invalid_arg "linear hardware IDs disagree with mapped coordinates";
  List.init rank (fun dimension ->
    let elements = List.nth mapping.elements_per_lane dimension
    and lanes=List.nth mapping.lanes_per_subgroup dimension
    and subgroups=List.nth mapping.subgroups_per_cta dimension
    and lane=List.nth hardware.lane dimension
    and register=List.nth hardware.register dimension
    and subgroup=List.nth hardware.subgroup dimension in
    if List.nth hardware.cta dimension <> 0 || lane < 0 || lane >= lanes ||
       register < 0 || register >= elements || subgroup < 0 || subgroup >= subgroups then
      invalid_arg "hardware coordinate is outside mapped CTA layout";
    let within=match mapping.lane_mapping with
      | Blocked -> lane * elements + register
      | Interleaved -> register * lanes + lane in
    subgroup * lanes * elements + within)

(* Allocation size includes alignment and any explicit per-row padding. *)
let shared_storage_bytes shape dtype layout = match layout with
  | Shared _ ->
      let bytes = Gpu_type.storage_bytes shape dtype in
      let alignment = 16 in
      if bytes > max_int - (alignment - 1) then
        invalid_arg "shared storage alignment overflows the host integer range";
      ((bytes + alignment - 1) / alignment) * alignment
  | Padded_shared { base; row_padding_bytes } ->
      if row_padding_bytes < 0 then invalid_arg "shared row padding must be nonnegative";
      let rank = List.length shape in
      if rank < 2 then invalid_arg "row-padded shared layouts require rank at least two";
      let extents=List.map (function
        | Gpu_type.Static n when n > 0 -> n
        | Gpu_type.Static _ -> invalid_arg "shared storage dimensions must be positive"
        | Gpu_type.Symbol _ | Gpu_type.Dynamic -> invalid_arg "shared storage size requires a fully static shape") shape in
      let fastest=List.hd base.order in
      let row_elements=List.nth extents fastest in
      let rows=Gpu_type.shape_elements shape / row_elements in
      let element_bytes=Gpu_type.dtype_storage_bytes dtype in
      if row_elements > max_int / element_bytes then
        invalid_arg "shared row byte size overflows the host integer range";
      let row_bytes=row_elements * element_bytes in
      if row_bytes > max_int - row_padding_bytes then
        invalid_arg "padded shared row size overflows the host integer range";
      let stride=row_bytes + row_padding_bytes in
      if rows > max_int / stride then invalid_arg "padded shared storage size overflows the host integer range";
      let bytes=rows * stride and alignment=16 in
      if bytes > max_int - (alignment - 1) then
        invalid_arg "shared storage alignment overflows the host integer range";
      ((bytes + alignment - 1) / alignment) * alignment
  | Register _ | Mapped_register _ | Global _ -> invalid_arg "shared storage size requires a shared layout"

let positive xs = List.for_all (fun n -> n > 0) xs

let static_extents shape =
  List.map (function
    | Gpu_type.Static n when n > 0 -> n
    | Gpu_type.Static _ -> invalid_arg "shape dimensions must be positive"
    | Gpu_type.Symbol _ | Gpu_type.Dynamic ->
        invalid_arg "layout addressing requires a fully static shape") shape

(* Row-major contiguous element strides: shape [R;C] -> [C;1]. *)
let contiguous_strides extents =
  List.fold_right (fun extent (stride, strides) ->
    (stride * extent, stride :: strides)) extents (1, []) |> snd

(* Shared layout `order` lists dimensions fastest-changing first. *)
let ordered_strides extents order =
  let rank = List.length extents in
  if not (valid_order rank order) then
    invalid_arg "shared layout order must be a permutation of tensor dimensions";
  let strides = Array.make rank 0 in
  let running = ref 1 in
  List.iter (fun dim ->
    strides.(dim) <- !running;
    running := !running * List.nth extents dim) order;
  Array.to_list strides

let linear_offset strides coordinates =
  if List.length strides <> List.length coordinates then
    invalid_arg "stride rank does not match coordinate rank";
  List.fold_left2 (fun acc stride coordinate -> acc + stride * coordinate) 0 strides coordinates

let memref_strides shape layout =
  let extents = static_extents shape in
  match layout with
  | None -> contiguous_strides extents
  | Some (Global { strides }) ->
      if List.length strides <> List.length extents then
        invalid_arg "global strides must match MemRef rank";
      if List.exists (fun s -> s <= 0) strides then
        invalid_arg "global strides must be positive";
      strides
  | Some (Shared { swizzle = No_swizzle; vector_width = 1; order })
  | Some (Shared { swizzle = Swizzle_32B | Swizzle_64B | Swizzle_128B;
                   vector_width = 1; order }) ->
      (* Swizzle is applied at PTX address emission time, not in the stride. *)
      ordered_strides extents order
  | Some (Shared _ | Padded_shared _) ->
      invalid_arg "padded shared addressing is not yet supported for tile movement"
  | Some (Register _ | Mapped_register _) ->
      invalid_arg "register layouts cannot address MemRef storage"

let validate_register_shape shape elements lanes subgroups order =
  let rank = List.length shape in
  if List.length elements <> rank || List.length lanes <> rank || List.length subgroups <> rank then
    Error "register layout factors must match the tensor rank"
  else if not (positive elements && positive lanes && positive subgroups) then
    Error "register layout factors must be positive"
  else if Option.fold ~none:false ~some:(fun order -> not (valid_order rank order)) order then
    Error "register layout order must be a permutation of tensor dimensions"
  else if List.exists2 (fun dim coverage -> match dim with
    | Gpu_type.Static extent -> extent <> coverage
    | Gpu_type.Symbol _ | Gpu_type.Dynamic -> false) shape
      (List.map2 (fun per_lane lane_count -> per_lane * lane_count)
        elements (List.map2 ( * ) lanes subgroups)) then
    Error "register layout factors must cover each static tile dimension exactly"
  else Ok ()

let validate shape layout =
  let rank = List.length shape in
  let dims_valid = List.for_all (function
    | Gpu_type.Static n -> n > 0
    | Gpu_type.Symbol name -> name <> ""
    | Gpu_type.Dynamic -> true) shape in
  if rank = 0 then Error "rank-zero layouts are not supported"
  else if not dims_valid then Error "shape dimensions must be positive or dynamic"
  else match layout with
    | Register r ->
        validate_register_shape shape r.elements_per_lane r.lanes_per_subgroup r.subgroups_per_cta (Some r.order)
    | Mapped_register m ->
        (match validate_register_shape shape m.elements_per_lane m.lanes_per_subgroup
          m.subgroups_per_cta None with
          | Error _ as error -> error
          | Ok () when not (valid_order rank m.lane_order && valid_order rank m.register_order &&
                            valid_order rank m.subgroup_order) ->
              Error "mapped register axis orders must be permutations of tensor dimensions"
          | Ok () -> Ok ())
    | Shared s | Padded_shared { base=s; _ } ->
        if not (valid_order rank s.order) then
          Error "shared layout order must be a permutation of tensor dimensions"
        else if s.vector_width <= 0 || (s.vector_width land (s.vector_width - 1)) <> 0 then
          Error "shared layout vector width must be a positive power of two"
        else if s.swizzle <> No_swizzle && rank < 2 then
          Error "shared-memory swizzling requires a rank-two-or-greater tile"
        else if (match layout with Padded_shared p -> p.row_padding_bytes < 0 | _ -> false) then
          Error "shared row padding must be nonnegative"
        else Ok ()
    | Global { strides } ->
        if List.length strides <> rank then
          Error "global strides must match the MemRef rank"
        else if List.exists (fun s -> s <= 0) strides then
          Error "global strides must be positive"
        else Ok ()

let string_of_swizzle = function
  | No_swizzle -> "none"
  | Swizzle_32B -> "32B"
  | Swizzle_64B -> "64B"
  | Swizzle_128B -> "128B"

(** XOR mask (in elements) for column swizzle: col' = col XOR (row AND mask). *)
let swizzle_element_mask = function
  | No_swizzle -> None
  | Swizzle_32B -> Some 7
  | Swizzle_64B -> Some 15
  | Swizzle_128B -> Some 31

let rec string_of_t = function
  | Register r ->
      Printf.sprintf "register<elements_per_lane=%s,lanes_per_subgroup=%s,subgroups_per_cta=%s,order=%s>"
        (String.concat "," (List.map string_of_int r.elements_per_lane))
        (String.concat "," (List.map string_of_int r.lanes_per_subgroup))
        (String.concat "," (List.map string_of_int r.subgroups_per_cta))
        (String.concat "," (List.map string_of_int r.order))
  | Mapped_register m ->
      Printf.sprintf "mapped_register<mode=%s,elements_per_lane=%s,lanes_per_subgroup=%s,subgroups_per_cta=%s,lane_order=%s,register_order=%s,subgroup_order=%s>"
        (match m.lane_mapping with Blocked -> "blocked" | Interleaved -> "interleaved")
        (String.concat "," (List.map string_of_int m.elements_per_lane))
        (String.concat "," (List.map string_of_int m.lanes_per_subgroup))
        (String.concat "," (List.map string_of_int m.subgroups_per_cta))
        (String.concat "," (List.map string_of_int m.lane_order))
        (String.concat "," (List.map string_of_int m.register_order))
        (String.concat "," (List.map string_of_int m.subgroup_order))
  | Shared s ->
      Printf.sprintf "shared<vector_width=%d,order=%s,swizzle=%s>"
        s.vector_width (String.concat "," (List.map string_of_int s.order))
        (string_of_swizzle s.swizzle)
  | Padded_shared p ->
      Printf.sprintf "padded_%s<row_padding_bytes=%d>"
        (string_of_t (Shared p.base)) p.row_padding_bytes
  | Global { strides } ->
      Printf.sprintf "global<strides=%s>"
        (String.concat "," (List.map string_of_int strides))
