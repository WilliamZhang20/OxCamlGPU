(* Thread and block coordinates, and the index maps the compiler derives from
   them. *)
open Typedtree
module K = Kernel_ast
open Typedtree_support
open Export_state

let coordinate : handler =
  on_positional (fun ctx -> function
    | [u] ->
        unit_arg ctx u;
        produce ~range:(0L, i32_max) ctx (fun d ->
          match ctx.op with
          | "thread_idx_x" -> K.Thread_idx_x d
          | "global_idx_x" -> K.Global_idx_x d
          | "block_idx_x" -> K.Block_idx_x d
          | _ -> K.Block_idx_y d)
    | _ -> arity ctx)

(* Which warpgroup the calling thread is in. 128 is the warpgroup width, a
   hardware constant, so the division belongs here. *)
let warpgroup_index : handler =
  on_positional (fun ctx -> function
    | [u] ->
        unit_arg ctx u;
        let { div; konst; _ } = index_arith ctx in
        (* Bound in emission order: arguments evaluate right to left, so
           passing these inline would reverse the instructions. *)
        let tid = thread_idx ctx in
        let width = konst 128 in
        Scalar (div tid width, full_range)
    | _ -> fail_here ctx "warpgroup_index expects ()")

(* Grouped CTA order. Row-major blockIdx makes every tile-row stream the whole
   of B, so B's DRAM traffic scales with the tile-row count. Walking [group]
   tile-rows before advancing along N keeps one A row-block resident across a
   group and cuts B's re-reads by that factor. This is the standard grouped
   bijection, so every tile is still covered exactly once, including when
   [tiles_m] is not a multiple of [group] and the last group is short. The
   group height is the author's tuning choice; the mapping is not. *)
let grouped_tile : handler =
 fun ctx xs ->
  match labelled ctx.exp.exp_loc ~required:["group"; "tiles_m"; "tiles_n"] xs with
  | [group; tiles_m; tiles_n], terminator ->
      Option.iter (unit_arg ctx) terminator;
      let group = num ctx group in
      let tiles_m = num ctx tiles_m in
      let tiles_n = num ctx tiles_n in
      let { add; sub; mul; div; rem; min_; _ } = index_arith ctx in
      let bx = value ctx (fun d -> K.Block_idx_x d) in
      let by = value ctx (fun d -> K.Block_idx_y d) in
      (* Linear tile id, then the group it falls in. *)
      let pid = add (mul by tiles_n) bx in
      let per_group = mul group tiles_n in
      let group_first = mul (div pid per_group) group in
      let group_rows = min_ (sub tiles_m group_first) group in
      let tile =
        if ctx.op = "grouped_tile_m" then add group_first (rem pid group_rows)
        else div (rem pid per_group) group_rows
      in
      Scalar (tile, full_range)
  | _ -> fail_here ctx "grouped_tile_m/n expect ~group ~tiles_m ~tiles_n"

let prims : (string * handler) list =
  List.map (fun op -> op, coordinate)
    ["thread_idx_x"; "global_idx_x"; "block_idx_x"; "block_idx_y"]
  @ [ "warpgroup_index", warpgroup_index
    ; "grouped_tile_m", grouped_tile
    ; "grouped_tile_n", grouped_tile ]
