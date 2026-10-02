open Gpu_mode
open Ir

(* A unique formal root cannot alias a distinct root. Derived pointers retain
   provenance, but equal roots and unknown roots remain conservatively aliasing. *)
let may_alias ?unique_roots kernel left right =
  match left.provenance, right.provenance with
  | Some left_root, Some right_root when left_root = right_root -> true
  | Some left_root, Some right_root ->
      let is_unique_root root =
        match unique_roots with
        | Some roots -> List.mem root roots
        | None ->
            List.exists (fun arg ->
              arg.value.id = root && arg.value.ownership = Unique)
              kernel.args
      in
      not (is_unique_root left_root || is_unique_root right_root)
  | _ -> true
