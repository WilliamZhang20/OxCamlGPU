(** Uniformity analysis for collective operations.

    [promote] rewrites an [If] to [If_uni] when every active lane of a warp
    takes the same arm: grid-, CTA-, or subgroup-uniform conditions, including
    a warp-aligned [thread_idx] compare ([tid < 256], [tid >= 256]). A one-lane
    compare ([tid = 256]) stays divergent. *)

val check : Ir.kernel -> (string * string) list
val promote : Ir.kernel -> Ir.kernel
