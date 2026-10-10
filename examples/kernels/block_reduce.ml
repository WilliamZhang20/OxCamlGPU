(* Block reduction, written the way you would write it in OCaml: a recursive
   tree fold parameterized by the combining function.

   Both of those are erased before PTX. [stride] halves by a compile-time
   constant, so the base-case test folds and the adapter traces only the taken
   arm; the recursion unrolls into log2(size) steps. [combine] is a
   compile-time function value, so it inlines at each step. The emitted kernel
   has no call, no closure and no recursion left in it.

   That is what makes one definition serve several reductions: [block_sum] and
   [block_max] below are the same fold under different operators. See
   docs/functions.md. *)
open Gpu_dsl

let block_reduce ~size ~combine =
  let schedule
      (src : float gpu_array @ aliased read)
      (dst : float gpu_array @ unique read_write) =
    let sm = Gpu.shared ~rows:1 ~cols:size () in
    let tid = Gpu.thread_idx_x () in
    Gpu.shared_store sm tid (Gpu.load src (Gpu.global_idx_x ()));
    Gpu.barrier_cta ();
    (* Halve the active range until nothing is left to combine. *)
    let rec fold stride =
      if stride = 0 then ()
      else begin
        if tid < stride then
          Gpu.shared_store sm tid
            (combine (Gpu.shared_load sm tid)
               (Gpu.shared_load sm (tid +: stride)));
        Gpu.barrier_cta ();
        fold (stride / 2)
      end
    in
    fold (size / 2);
    (* One result per block. *)
    if tid = 0 then Gpu.store dst (Gpu.block_idx_x ()) (Gpu.shared_load sm 0)
  in
  schedule

let block_sum =
  block_reduce ~size:256 ~combine:(fun a b -> a +. b)
[@@gpu.threads 256]

let block_max =
  block_reduce ~size:256 ~combine:(fun a b -> if a > b then a else b)
[@@gpu.threads 256]
