type swizzle = No_swizzle | Swizzle_32B | Swizzle_64B | Swizzle_128B

type register = {
  (* A target-independent distribution: each logical dimension is split
     across per-lane elements, lanes in a subgroup, and subgroups in a CTA. *)
  elements_per_lane : int list;
  lanes_per_subgroup : int list;
  subgroups_per_cta : int list;
  order : int list;
}

type shared = {
  vector_width : int;
  order : int list;
  swizzle : swizzle;
}

type t = Register of register | Shared of shared

(* A logical coordinate is decomposed into coordinates at each hardware
   level. The per-dimension vectors keep this mapping compositional: target
   lowering may later linearize or swizzle each level for a specific GPU. *)
type hardware_coordinate = {
  cta : int list;
  subgroup : int list;
  lane : int list;
  register : int list;
}

let logical_to_hardware register logical =
  let factors = List.combine register.elements_per_lane
      (List.combine register.lanes_per_subgroup register.subgroups_per_cta) in
  if List.length logical <> List.length factors then
    invalid_arg "logical coordinate rank does not match register layout";
  let parts = List.map2 (fun coordinate (elements, (lanes, subgroups)) ->
    if coordinate < 0 then invalid_arg "logical coordinates must be nonnegative";
    let per_lane, distributed = coordinate mod elements, coordinate / elements in
    let lane = distributed mod lanes and cta = distributed / lanes in
    let subgroup = cta mod subgroups and cta = cta / subgroups in
    per_lane, lane, subgroup, cta) logical factors in
  { register=List.map (fun (x,_,_,_) -> x) parts;
    lane=List.map (fun (_,x,_,_) -> x) parts;
    subgroup=List.map (fun (_,_,x,_) -> x) parts;
    cta=List.map (fun (_,_,_,x) -> x) parts }

let hardware_to_logical layout hardware =
  let rank = List.length layout.elements_per_lane in
  if List.exists (fun xs -> List.length xs <> rank)
      [hardware.cta;hardware.subgroup;hardware.lane;hardware.register] then
    invalid_arg "hardware coordinate rank does not match register layout";
  List.init rank (fun dimension ->
    let elements=List.nth layout.elements_per_lane dimension
    and lanes=List.nth layout.lanes_per_subgroup dimension
    and subgroups=List.nth layout.subgroups_per_cta dimension in
    let register=List.nth hardware.register dimension
    and lane=List.nth hardware.lane dimension
    and subgroup=List.nth hardware.subgroup dimension
    and cta=List.nth hardware.cta dimension in
    if register < 0 || register >= elements || lane < 0 || lane >= lanes ||
       subgroup < 0 || subgroup >= subgroups || cta <> 0 then
      invalid_arg "hardware coordinate is outside this CTA register layout";
    (((subgroup * lanes + lane) * elements) + register))

(* Shared layouts currently preserve the logical extent. Swizzle-specific
   padding can be added here without changing launch-contract construction. *)
let shared_storage_bytes shape dtype layout = match layout with
  | Shared _ ->
      let bytes = Gpu_type.storage_bytes shape dtype in
      let alignment = 16 in
      if bytes > max_int - (alignment - 1) then
        invalid_arg "shared storage alignment overflows the host integer range";
      ((bytes + alignment - 1) / alignment) * alignment
  | Register _ -> invalid_arg "shared storage size requires a shared layout"

let valid_order rank order =
  List.length order = rank
  && List.sort compare order = List.init rank Fun.id

let positive xs = List.for_all (fun n -> n > 0) xs

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
        if List.length r.elements_per_lane <> rank
           || List.length r.lanes_per_subgroup <> rank
           || List.length r.subgroups_per_cta <> rank then
          Error "register layout factors must match the tensor rank"
        else if not (positive r.elements_per_lane && positive r.lanes_per_subgroup && positive r.subgroups_per_cta) then
          Error "register layout factors must be positive"
        else if not (valid_order rank r.order) then
          Error "register layout order must be a permutation of tensor dimensions"
        else if List.exists2 (fun dim coverage -> match dim with
          | Gpu_type.Static extent -> extent <> coverage
          | Gpu_type.Symbol _ | Gpu_type.Dynamic -> false)
            shape (List.map2 (fun per_lane lanes -> per_lane * lanes)
              r.elements_per_lane
              (List.map2 ( * ) r.lanes_per_subgroup r.subgroups_per_cta)) then
          Error "register layout factors must cover each static tile dimension exactly"
        else Ok ()
    | Shared s ->
        if not (valid_order rank s.order) then
          Error "shared layout order must be a permutation of tensor dimensions"
        else if s.vector_width <= 0 || (s.vector_width land (s.vector_width - 1)) <> 0 then
          Error "shared layout vector width must be a positive power of two"
        else if s.swizzle <> No_swizzle && rank < 2 then
          Error "shared-memory swizzling requires a rank-two-or-greater tile"
        else Ok ()

let string_of_swizzle = function
  | No_swizzle -> "none"
  | Swizzle_32B -> "32B"
  | Swizzle_64B -> "64B"
  | Swizzle_128B -> "128B"

let string_of_t = function
  | Register r ->
      Printf.sprintf "register<elements_per_lane=%s,lanes_per_subgroup=%s,subgroups_per_cta=%s,order=%s>"
        (String.concat "," (List.map string_of_int r.elements_per_lane))
        (String.concat "," (List.map string_of_int r.lanes_per_subgroup))
        (String.concat "," (List.map string_of_int r.subgroups_per_cta))
        (String.concat "," (List.map string_of_int r.order))
  | Shared s ->
      Printf.sprintf "shared<vector_width=%d,order=%s,swizzle=%s>"
        s.vector_width (String.concat "," (List.map string_of_int s.order))
        (string_of_swizzle s.swizzle)
