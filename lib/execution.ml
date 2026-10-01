(* Execution scopes describe which GPU agents participate in an operation.
   They are independent of memory address spaces and data layouts. *)
type level = Grid | Cta | Subgroup | Lane | Warpgroup

let parent = function
  | Grid -> None
  | Cta -> Some Grid
  | Subgroup -> Some Cta
  | Lane -> Some Subgroup
  (* Warpgroup is an optional overlapping hardware grouping, not a mandatory
     parent in the semantic execution hierarchy. *)
  | Warpgroup -> None
