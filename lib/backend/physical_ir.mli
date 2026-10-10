(** Physical register mapping for a target kernel. *)

type register = {
  index : int;
  dtype : Gpu_type.dtype option;
  value : Ir.value;
}

type value = {
  semantic : Ir.value;
  registers : register list;
  layout_mapping : Layout.mapped_register option;
}

type kernel = {
  target : Ptx_ir.kernel;
  values : (int * value) list;
}

type tile_slot_address =
  | Linear of { base : int; lane_stride : int }
  | Rank2 of {
      tile_cols : int;
      row_offset : int;
      col_offset : int;
      row_stride : int;
    }

val physicalize : Ptx_ir.kernel -> kernel
val lookup : kernel -> Ir.value -> value option
val coordinate : kernel -> Ir.value -> int list -> Layout.hardware_coordinate
