(** Fuse consecutive scalar f32 stores into vector stores.

    A run of [Store_f32] becomes one [Store_f32x2] or [Store_f32x4] when the
    element indices are provably consecutive and the first index is provably a
    multiple of the vector width, which PTX requires for alignment. Indices
    are compared through an affine form over structurally keyed atoms, and
    alignment through a congruence, so neither a prior CSE pass nor
    syntactically identical index expressions are needed.

    Stores that cannot be proved consecutive and aligned are left alone, so
    this is always safe to run. *)

val fuse : Ir.kernel -> Ir.kernel
