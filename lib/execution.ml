(* Execution scopes describe which GPU agents participate in an operation.
   They are independent of memory address spaces and data layouts. *)
type level = Grid | Cta | Warpgroup | Warp | Lane

let parent = function
  | Grid -> None
  | Cta -> Some Grid
  | Warpgroup -> Some Cta
  | Warp -> Some Warpgroup
  | Lane -> Some Warp
