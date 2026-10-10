(** Data distribution and address layouts. Independent of value uniformity
    and of OxCaml locality/ownership modes. *)

type swizzle = No_swizzle | Swizzle_32B | Swizzle_64B | Swizzle_128B

type register = {
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
  lane_order : int list;
  register_order : int list;
  subgroup_order : int list;
}

type shared = {
  vector_width : int;
  order : int list;
  swizzle : swizzle;
}

type padded_shared = { base : shared; row_padding_bytes : int }

type global = { strides : int list }

type t =
  | Register of register
  | Mapped_register of mapped_register
  | Shared of shared
  | Padded_shared of padded_shared
  | Global of global

type hardware_coordinate = {
  cta : int list;
  subgroup : int list;
  lane : int list;
  register : int list;
  lane_id : int;
  register_id : int;
  subgroup_id : int;
}

val mapped_to_hardware : mapped_register -> int list -> hardware_coordinate
val logical_to_hardware : register -> int list -> hardware_coordinate
val hardware_to_logical : register -> hardware_coordinate -> int list
val mapped_hardware_to_logical : mapped_register -> hardware_coordinate -> int list

val shared_storage_bytes : Gpu_type.dim list -> Gpu_type.dtype -> t -> int
val static_extents : Gpu_type.dim list -> int list
val linear_offset : int list -> int list -> int
val memref_strides : Gpu_type.dim list -> t option -> int list

val validate : Gpu_type.dim list -> t -> (unit, string) result

(** XOR mask (in elements) for column swizzle: [col' = col XOR (row AND mask)]. *)
val swizzle_element_mask : swizzle -> int option

val string_of_t : t -> string
