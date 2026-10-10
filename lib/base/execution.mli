(** Execution scopes: which GPU agents participate in an operation. *)

type level = Grid | Cta | Subgroup | Lane | Warpgroup

val parent : level -> level option
