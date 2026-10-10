(** Tile shapes and launch contract for the Hopper TMA+WGMMA GEMM emitter. *)

type tile_config = {
  bm : int;
  bn : int;
  bk : int;
  stages : int;
  producers : int;
}

val warp_groups : tile_config -> int
val consumers : tile_config -> int
val threads : tile_config -> int

val validate : tile_config -> tile_config
val string_of_config : tile_config -> string
val parse_config : string -> tile_config

(** Autotune catalog (preferred configs first). *)
val tile_catalog : tile_config list

(** Default CTA tile for square mid-size TF32 GEMM. *)
val hopper_f32 : tile_config

val choose_config : ?m:int -> ?n:int -> ?k:int -> unit -> tile_config

(** OxCaml entry name for a catalog or [choose_config] tile. *)
val kernel_binding : tile_config -> string
