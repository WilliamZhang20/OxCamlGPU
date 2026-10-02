open Gpu_dsl
module Device = Gpu
type float_buffer = float gpu_array

let guarded_saxpy :
  float_buffer @ aliased read ->
  float_buffer @ unique read_write -> float -> int -> unit = fun x y a n ->
  let i = Device.global_idx_x () in
  if i < n then begin
    let xi = Device.load x i in
    let contribution = if xi > 0. then (let scaled = a *. xi in scaled) else 0. in
    Device.store y i (contribution +. Device.load y i)
  end

(* The RHS must never load the null buffer passed by the GPU harness. *)
let short_circuit :
  float gpu_array @ aliased read -> float gpu_array @ unique read_write -> bool -> int -> unit = fun x y gate n ->
  let i = Gpu.global_idx_x () in
  if i < n then begin
    let a = gate && Gpu.load x i > 0. in
    let b = not gate || Gpu.load x i > 0. in
    let answer = if a then 99. else if b then 7. else 88. in
    Gpu.store y i answer
  end

let numeric :
  float gpu_array @ unique read_write -> int -> unit = fun y n ->
  let i = Gpu.global_idx_x () in
  if i < n then begin
    let wrapped = Int32.add 2147483647l 1l in
    let product = Int32.mul (Int32.sub 5l 2l) 7l in
    let ordinary = (2 + 3) * 4 - 1 in
    let a = wrapped < 0l && product = 21l in
    let result = if a && ordinary = 19 then 1. else 0. in
    Gpu.store y i result
  end

let float_compare :
  float gpu_array @ aliased read -> float gpu_array @ unique read_write -> int -> unit = fun x y n ->
  let i = Gpu.global_idx_x () in
  if i < n then begin
    let v = Gpu.load x i in
    let result = if v <> v then 3. else if v = 0. then 2. else if v < 0. then 1. else 0. in
    Gpu.store y i result
  end

let uniform_branch :
  float gpu_array @ unique read_write -> bool -> int -> unit = fun y choose n ->
  let i = Gpu.global_idx_x () in
  if i < n then Gpu.store y i (if choose then 9. else 4.)

let joined_reduction : float gpu_array @ unique write -> unit = fun y ->
  let lane = Gpu.thread_idx_x () in
  let value = if lane < 16 then 1. else 2. in
  Gpu.store_grid_leader y (Gpu.warp_sum_f32 value)

let divergent_reduction : float gpu_array @ unique write -> unit = fun y ->
  let lane = Gpu.thread_idx_x () in
  let value = if lane < 16 then Gpu.warp_sum_f32 1. else 2. in
  Gpu.store_grid_leader y value

let rounding : float gpu_array @ unique read_write -> float -> unit = fun y a ->
  let value = a *. a +. (-1.0000002384185791015625) in
  Gpu.store y 0 value
