type addr_space = Global | Shared | Local

type ty = I32 | F32 | Unit | Ptr of ty * addr_space

let rec string_of_ty = function
  | I32 -> "i32"
  | F32 -> "f32"
  | Unit -> "unit"
  | Ptr (t, space) ->
      let space = match space with Global -> "global" | Shared -> "shared" | Local -> "local" in
      Printf.sprintf "ptr<%s,%s>" (string_of_ty t) space
