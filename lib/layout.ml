type swizzle = No_swizzle | Swizzle_32B | Swizzle_64B | Swizzle_128B

type register = {
  size_per_thread : int list;
  threads_per_warp : int list;
  warps_per_cta : int list;
  order : int list;
}

type shared = {
  vector_width : int;
  order : int list;
  swizzle : swizzle;
}

type t = Register of register | Shared of shared

let valid_order rank order =
  List.length order = rank
  && List.sort compare order = List.init rank Fun.id

let positive xs = List.for_all (fun n -> n > 0) xs

let validate shape layout =
  let rank = List.length shape in
  let dims_valid = List.for_all (function
    | Gpu_type.Static n -> n > 0
    | Gpu_type.Symbol name -> name <> ""
    | Gpu_type.Dynamic -> true) shape in
  if rank = 0 then Error "rank-zero layouts are not supported"
  else if not dims_valid then Error "shape dimensions must be positive or dynamic"
  else match layout with
    | Register r ->
        if List.length r.size_per_thread <> rank
           || List.length r.threads_per_warp <> rank
           || List.length r.warps_per_cta <> rank then
          Error "register layout factors must match the tensor rank"
        else if not (positive r.size_per_thread && positive r.threads_per_warp && positive r.warps_per_cta) then
          Error "register layout factors must be positive"
        else if not (valid_order rank r.order) then
          Error "register layout order must be a permutation of tensor dimensions"
        else if List.exists2 (fun dim coverage -> match dim with
          | Gpu_type.Static extent -> extent <> coverage
          | Gpu_type.Symbol _ | Gpu_type.Dynamic -> false)
            shape (List.map2 (fun per_thread lanes -> per_thread * lanes)
              r.size_per_thread
              (List.map2 ( * ) r.threads_per_warp r.warps_per_cta)) then
          Error "register layout factors must cover each static tile dimension exactly"
        else Ok ()
    | Shared s ->
        if not (valid_order rank s.order) then
          Error "shared layout order must be a permutation of tensor dimensions"
        else if s.vector_width <= 0 || (s.vector_width land (s.vector_width - 1)) <> 0 then
          Error "shared layout vector width must be a positive power of two"
        else if s.swizzle <> No_swizzle && rank < 2 then
          Error "shared-memory swizzling requires a rank-two-or-greater tile"
        else Ok ()

let string_of_swizzle = function
  | No_swizzle -> "none"
  | Swizzle_32B -> "32B"
  | Swizzle_64B -> "64B"
  | Swizzle_128B -> "128B"

let string_of_t = function
  | Register r ->
      Printf.sprintf "register<size_per_thread=%s,threads_per_warp=%s,warps_per_cta=%s,order=%s>"
        (String.concat "," (List.map string_of_int r.size_per_thread))
        (String.concat "," (List.map string_of_int r.threads_per_warp))
        (String.concat "," (List.map string_of_int r.warps_per_cta))
        (String.concat "," (List.map string_of_int r.order))
  | Shared s ->
      Printf.sprintf "shared<vector_width=%d,order=%s,swizzle=%s>"
        s.vector_width (String.concat "," (List.map string_of_int s.order))
        (string_of_swizzle s.swizzle)
