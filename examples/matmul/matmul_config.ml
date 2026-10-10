(* Tile shapes and launch contract for the Hopper TMA+WGMMA GEMM, and the
   autotune catalog. Every definition here is part of the interface the
   emitter and the autotune driver use, so there is no signature to narrow.

   consumers = 128 × (bm/64) warp groups (WGMMA m64).
   producers = TMA elect warps (multiple of 32; typically 32).
   stages = software pipeline depth (≥2). *)

type tile_config =
  { bm : int
  ; bn : int
  ; bk : int
  ; stages : int
  ; producers : int
  }

let warp_groups (c : tile_config) = c.bm / 64
let consumers (c : tile_config) = 128 * warp_groups c
let threads (c : tile_config) = consumers c + c.producers

let validate (c : tile_config) =
  let wgmma_n_ok n = n >= 8 && n <= 256 && n mod 8 = 0 in
  if c.bm mod 64 <> 0 || c.bm < 64 then
    invalid_arg "BM must be a positive multiple of 64 (WGMMA m64)";
  if not (wgmma_n_ok c.bn) then
    invalid_arg "BN must be in {8,16,...,256} for WGMMA";
  if c.bk mod 32 <> 0 || c.bk < 32 || c.bk > 32 then
    invalid_arg "BK must be 32 (f32 TMA 128B swizzle box0 ≤ 32)";
  if c.stages < 2 then
    invalid_arg "pipeline stages must be ≥ 2";
  if c.producers <= 0 || c.producers mod 32 <> 0 then
    invalid_arg "producers must be a positive multiple of warp size (32)";
  if threads c > 1024 then
    invalid_arg "CTA threads (consumers+producers) must be ≤ 1024";
  c

let string_of_config (c : tile_config) =
  Printf.sprintf
    "bm=%d,bn=%d,bk=%d,stages=%d,producers=%d,threads=%d"
    c.bm c.bn c.bk c.stages c.producers (threads c)

let parse_config s =
  let tbl = Hashtbl.create 8 in
  String.split_on_char ',' s
  |> List.iter (fun part ->
      match String.split_on_char '=' part with
      | [k; v] -> Hashtbl.replace tbl (String.trim k) (String.trim v)
      | _ -> invalid_arg ("bad config fragment: " ^ part));
  let get k =
    match Hashtbl.find_opt tbl k with
    | Some v -> int_of_string v
    | None -> invalid_arg ("missing config key: " ^ k)
  in
  let stages =
    match Hashtbl.find_opt tbl "stages" with
    | Some v -> int_of_string v
    | None -> 3
  in
  (* Unknown keys such as legacy tm/tn are ignored. WGMMA owns the mapping. *)
  validate
    { bm = get "bm"
    ; bn = get "bn"
    ; bk = get "bk"
    ; stages
    ; producers = get "producers"
    }

(* Prefer wide-N TF32 WGMMA at stages=3, which is what [choose_config] picks.
   The stages=2 entries are kept so an autotune sweep can still measure them,
   but they are dominated on an H100 since the K loop spends one stage of
   producer runway keeping a WGMMA group in flight. A+B stay under an SM's
   228KiB of shared memory, which the launch requests dynamically. *)
let tile_catalog =
  List.map validate
    [ { bm = 128; bn = 256; bk = 32; stages = 4; producers = 32 }
    ; { bm = 128; bn = 256; bk = 32; stages = 3; producers = 32 }
    ; { bm = 128; bn = 256; bk = 32; stages = 2; producers = 32 }
    ; { bm = 128; bn = 128; bk = 32; stages = 3; producers = 32 }
    ; { bm = 128; bn = 128; bk = 32; stages = 2; producers = 32 }
    ; { bm = 64; bn = 256; bk = 32; stages = 3; producers = 32 }
    ; { bm = 256; bn = 128; bk = 32; stages = 3; producers = 32 }
    ]

let hopper_f32 = List.hd tile_catalog

(* Names of the OxCaml specializations in examples/matmul/matmul_tiled.ml.
   [matmul_tiled] stays the default 128×256 stages=3 entry (H100 harness).
   [choose_config] only asks for stages=3; the stages=2 bindings remain
   reachable through [parse_config] and an autotune sweep. *)
let kernel_binding (c : tile_config) =
  match c.bm, c.bn, c.bk, c.stages, c.producers with
  | 128, 256, 32, 4, 32 -> "matmul_tiled"
  | 128, 256, 32, 3, 32 -> "matmul_bm128_bn256_s3"
  | 128, 256, 32, 2, 32 -> "matmul_bm128_bn256_s2"
  | 128, 128, 32, 3, 32 -> "matmul_bm128_bn128_s3"
  | 128, 128, 32, 2, 32 -> "matmul_bm128_bn128_s2"
  | 64, 256, 32, 3, 32 -> "matmul_bm64_bn256_s3"
  | 64, 256, 32, 2, 32 -> "matmul_bm64_bn256_s2"
  | 256, 128, 32, 3, 32 -> "matmul_bm256_bn128_s3"
  | 256, 128, 32, 2, 32 -> "matmul_bm256_bn128_s2"
  | _ -> invalid_arg ("no OxCaml kernel for " ^ string_of_config c)

let choose_config ?(m = 4096) ?(n = 4096) ?k:(_k = 4096) () =
  (* Deeper is better, because the K loop keeps one WGMMA group in flight and
     releases the previous stage, which spends one stage of producer runway.
     Measured on an H100 at 128x256x32: stages=2 is far back, at 59% of cuBLAS
     at 4096 and 61% at 8192. Past that the gain is real but smaller than the
     headline suggests. At 4096 stages=3 holds 379-381 TFLOPS and stages=4
     386-391, so four stages are worth about two points and that is
     repeatable. At 8192 the two overlap, 390-403 against 372-405 over three
     alternating rounds, so the choice there is within run-to-run noise and
     four stages are kept only because they do not cost anything. Four stages
     need 197 KiB of shared memory, inside an SM's 228 KiB, and were only
     reachable once the barriers became an indexed set rather than a fixed
     three-deep ladder.

     Only the 128x256 shape has been measured at four stages, so the rarer
     aspect-ratio tiles stay at three. K does not steer the tile choice; the
     M/N aspect ratio does, and the parameter stays for callers and for a
     future heuristic that wants it. *)
  if m >= 2 * n then
    validate { bm = 256; bn = 128; bk = 32; stages = 3; producers = 32 }
  else if m <= 1024 && n <= 1024 && m < 2 * n && n < 2 * m then
    validate { bm = 64; bn = 256; bk = 32; stages = 3; producers = 32 }
  else
    validate { bm = 128; bn = 256; bk = 32; stages = 4; producers = 32 }
