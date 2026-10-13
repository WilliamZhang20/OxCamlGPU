(* Structured control flow: branch scope and yields, collective uniformity,
   shared-memory initialization at joins, and what the optimizer may carry
   across a branch. Each scenario below is one named test so a failure says
   which property broke.

   Run with [--ptx] to dump the branch kernel's PTX instead of asserting; the
   hardware harness assembles that. *)

open Gpu_type
open Ir

(* --- assertions ------------------------------------------------------- *)

let ok kernel = Verifier.verify_exn kernel

let rejects code kernel =
  match Verifier.verify_kernel kernel with
  | Error errors when List.exists (fun e -> e.Verifier.code = code) errors -> ()
  | Error errors ->
      failwith
        (Printf.sprintf "expected %s, got %s" code
           (String.concat "," (List.map (fun e -> e.Verifier.code) errors)))
  | Ok () -> failwith ("expected " ^ code ^ ", but the kernel verified")

let contains text fragment =
  let width = String.length fragment in
  let rec loop at =
    at + width <= String.length text
    && (String.sub text at width = fragment || loop (at + 1))
  in
  loop 0

let count_loads body =
  List.length
    (List.filter (function Load_f32 _ -> true | _ -> false) (Ir.flatten body))

(* --- fixture ---------------------------------------------------------- *)

(* Value ids only have to be distinct inside one kernel, so a counter is safer
   than hand-picked numbers: adding a case cannot collide with an existing
   one. The two places an id carries meaning say so where they are defined. *)
let fresh =
  let next = ref 0 in
  fun ?permission ?layout ty ->
    incr next;
    make_value ?permission ?layout !next ty

let region ?yield body = { Ir.body; yield }
let writable = Gpu_mode.Read_write

let output = fresh ~permission:writable (Ptr (F32, Global))
let index = fresh I32
let limit = fresh I32
let cond = fresh Bool
let a = fresh F32
let b = fresh F32
let selected = fresh F32
let element = fresh ~permission:writable (Ptr (F32, Global))
let yes = fresh Bool
let no = fresh Bool
let lane_value = fresh F32
let reduced = fresh F32

(* tid < 16: not a multiple of the warp width, so the split is divergent. *)
let divergent_prefix =
  [ Thread_idx_x index; Const_i32 (limit, 16); Compare (cond, Lt, index, limit) ]

let select_branch =
  If
    ( Some selected, cond
    , region ~yield:a [ Const_f32 (a, 1.) ]
    , region ~yield:b [ Const_f32 (b, 2.) ] )

let branch_kernel =
  { name = "branch_ir"
  ; args = [ { name = "output"; value = output } ]
  ; body =
      divergent_prefix
      @ [ select_branch
        ; Gep_f32 (element, output, index)
        ; Store_f32 (element, selected)
        ]
  ; threads_per_cta = None
  }

let body instructions = { branch_kernel with body = instructions }
let ok_body instructions = ok (body instructions)
let rejects_body code instructions = rejects code (body instructions)
let ptx_of instructions = Compiler.compile_ptx (body instructions)

(* --- scenarios -------------------------------------------------------- *)

let branch_lowers_to_a_join () =
  ok branch_kernel;
  let emitted = Compiler.compile_ptx branch_kernel in
  (* The join writes the yielded value into the [If]'s own register. Ask the
     value for its name rather than writing %f6, which pinned this assertion
     to one particular id. *)
  let join = Printf.sprintf "mov.f32 %%f%d" selected.id in
  if not (contains emitted "bra L" && contains emitted join) then
    failwith ("missing target branch or join " ^ join);
  if contains emitted "bra.uni" then
    failwith "tid < 16 must stay a divergent branch"

let branch_scope_is_enforced () =
  (* A value defined inside an arm does not escape it. *)
  rejects_body "E_UNDEFINED" (divergent_prefix @ [ select_branch; Store_f32 (output, a) ]);
  (* An [If] with a result needs a yield from both arms, of that type. *)
  rejects_body "E_YIELD_TYPE"
    (divergent_prefix
     @ [ If (Some selected, cond, region [ Const_f32 (a, 1.) ],
             region ~yield:b [ Const_f32 (b, 2.) ]) ]);
  (* An arm cannot yield the result it is defining. *)
  rejects_body "E_UNDEFINED"
    (divergent_prefix
     @ [ If (Some selected, cond, region ~yield:selected [],
             region ~yield:b [ Const_f32 (b, 2.) ]) ]);
  (* Both arms defining the same SSA value is a redefinition, not a join. *)
  rejects_body "E_REDEFINITION"
    (divergent_prefix
     @ [ If (Some selected, cond, region ~yield:a [ Const_f32 (a, 1.) ],
             region ~yield:a [ Const_f32 (a, 2.) ]) ]);
  rejects_body "E_CONDITION_TYPE"
    (divergent_prefix @ [ If (None, index, region [], region []) ]);
  rejects_body "E_REGION_RETURN"
    (divergent_prefix @ [ If (None, cond, region [ Return None ], region []) ]);
  (* A join must not strengthen either arm's modal facts. *)
  let forged = { selected with domain_portability = Gpu_mode.Domain_portable } in
  rejects_body "E_YIELD_FACTS"
    (divergent_prefix
     @ [ If (Some forged, cond, region ~yield:a [ Const_f32 (a, 1.) ],
             region ~yield:b [ Const_f32 (b, 2.) ]) ])

let collectives_need_uniform_control () =
  (* A warp reduction inside a divergent arm has an incomplete warp. *)
  rejects_body "E_DIVERGENT_COLLECTIVE"
    (divergent_prefix
     @ [ Const_f32 (lane_value, 1.)
       ; If (None, cond, region [ Warp_sum_f32 (reduced, lane_value) ], region [])
       ]);
  (* Nesting a uniform branch inside a divergent one does not recover it. *)
  rejects_body "E_DIVERGENT_COLLECTIVE"
    (divergent_prefix
     @ [ Const_bool (yes, true)
       ; If (None, cond,
             region [ If (None, yes, region [ Barrier Execution.Cta ], region []) ],
             region [])
       ]);
  (* Nor does laundering the condition through a join: [choice] is divergent
     because the branch that produced it was. *)
  let choice = fresh Bool in
  rejects_body "E_DIVERGENT_COLLECTIVE"
    (divergent_prefix
     @ [ If (Some choice, cond, region ~yield:yes [ Const_bool (yes, true) ],
             region ~yield:no [ Const_bool (no, false) ])
       ; If (None, choice, region [ Barrier Execution.Cta ], region [])
       ]);
  (* A branch on a compile-time constant is uniform, so a barrier is fine. *)
  ok_body
    [ Const_bool (yes, true)
    ; If (None, yes, region [ Barrier Execution.Cta ], region [])
    ];
  (* And a reduction after the join is fine: every lane reaches it. *)
  ok_body
    (divergent_prefix
     @ [ select_branch
       ; Warp_sum_f32 (reduced, selected)
       ; Store_f32 (output, reduced)
       ])

(* A lane remap is a collective: it moves values between lanes. *)
let remap_is_collective () =
  let one_warp lane_mapping =
    Layout.Mapped_register
      { elements_per_lane = [ 1 ]
      ; lanes_per_subgroup = [ 32 ]
      ; subgroups_per_cta = [ 1 ]
      ; lane_mapping
      ; lane_order = [ 0 ]
      ; register_order = [ 0 ]
      ; subgroup_order = [ 0 ]
      }
  in
  let shape = [ Static 32 ] in
  let buffer = fresh ~permission:writable (MemRef (shape, Float32, Global)) in
  let source = fresh ~layout:(one_warp Layout.Blocked) (Tensor (shape, Float32)) in
  let remapped =
    fresh ~layout:(one_warp Layout.Interleaved) (Tensor (shape, Float32))
  in
  rejects "E_DIVERGENT_COLLECTIVE"
    { branch_kernel with
      args = branch_kernel.args @ [ { name = "remap_buf"; value = buffer } ]
    ; body =
        divergent_prefix
        @ [ Load_tensor (source, buffer)
          ; If (None, cond, region [ Remap_tensor (remapped, source) ], region [])
          ]
    }

let shared_tiles_join_at_barriers () =
  let shape = [ Static 32 ] in
  let registers =
    Layout.Register
      { elements_per_lane = [ 1 ]; lanes_per_subgroup = [ 32 ]
      ; subgroups_per_cta = [ 1 ]; order = [ 0 ] }
  in
  let tile_layout =
    Layout.Shared { vector_width = 1; order = [ 0 ]; swizzle = Layout.No_swizzle }
  in
  let smem =
    fresh ~permission:writable ~layout:tile_layout (MemRef (shape, Float32, Shared))
  in
  let source = fresh (MemRef (shape, Float32, Global)) in
  let tile = fresh ~layout:registers (Tensor (shape, Float32)) in
  let loaded = fresh ~layout:registers (Tensor (shape, Float32)) in
  let shared_kernel instructions =
    { branch_kernel with
      args = branch_kernel.args @ [ { name = "src"; value = source } ]
    ; body = instructions }
  in
  let setup = [ Const_bool (yes, true); Shared_alloc smem; Load_tensor (tile, source) ] in
  (* SPMD If: a store in either arm initializes the tile for the whole CTA. *)
  ok (shared_kernel
        (setup
         @ [ If (None, yes, region [ Store_tensor (smem, tile) ], region [])
           ; Barrier Execution.Cta
           ; Load_tensor (loaded, smem)
           ]));
  (* Neither arm storing leaves the tile undefined at the read. *)
  rejects "E_SHARED_UNINITIALIZED"
    (shared_kernel
       (setup
        @ [ If (None, yes, region [], region [])
          ; Barrier Execution.Cta
          ; Load_tensor (loaded, smem)
          ]));
  (* A barrier inside one arm does not synchronize the CTA. *)
  rejects "E_SHARED_SYNC"
    (shared_kernel
       (setup
        @ [ Store_tensor (smem, tile)
          ; If (None, yes, region [ Barrier Execution.Cta ], region [])
          ; Load_tensor (loaded, smem)
          ]))

let branch_store_invalidates_the_load_cache () =
  let before = fresh F32 and after = fresh F32 in
  let kernel =
    body
      (divergent_prefix
       @ [ Load_f32 (before, output)
         ; If (None, cond, region [ Store_f32 (output, before) ], region [])
         ; Load_f32 (after, output)
         ])
  in
  if count_loads (Optimizer.optimize kernel).body <> 2 then
    failwith "a store inside a branch left a stale load cache"

let inherited_pointers_stay_distinct () =
  (* A branch optimizer cannot treat parent-defined derived pointers as their
     identical root address when its local address map starts empty. The ids
     here are load-bearing: a pointer is its own root when id = provenance. *)
  let root = make_value ~provenance:(Some 500) 500 (Ptr (F32, Global)) in
  let first = make_value ~provenance:(Some 500) 501 (Ptr (F32, Global)) in
  let second = make_value ~provenance:(Some 500) 502 (Ptr (F32, Global)) in
  let zero = fresh I32 and one = fresh I32 in
  let loaded_a = fresh F32 and loaded_b = fresh F32 in
  let kernel =
    { branch_kernel with
      args = [ { name = "input"; value = root } ]
    ; body =
        [ Const_bool (yes, true)
        ; Const_i32 (zero, 0)
        ; Const_i32 (one, 1)
        ; Gep_f32 (first, root, zero)
        ; Gep_f32 (second, root, one)
        ; If (None, yes,
              region [ Load_f32 (loaded_a, first); Load_f32 (loaded_b, second) ],
              region [])
        ]
    }
  in
  if count_loads (Optimizer.optimize kernel).body <> 2 then
    failwith "distinct inherited pointers were merged"

(* 256 is a multiple of the warp width, so a tid split on it is
   subgroup-uniform: a warp sum inside is legal, a CTA barrier is not, because
   uniformity within a warp says nothing about the other warps. *)
let warp_aligned_splits () =
  let wide_limit = fresh I32 in
  let wide = fresh Bool and elect = fresh Bool in
  let shifted = fresh I32 and one = fresh I32 and shifted_cond = fresh Bool in
  let split comparison result =
    [ Thread_idx_x index
    ; Const_i32 (wide_limit, 256)
    ; Compare (result, comparison, index, wide_limit)
    ]
  in
  ok_body
    (split Lt wide
     @ [ Const_f32 (lane_value, 1.)
       ; If (None, wide, region [ Warp_sum_f32 (reduced, lane_value) ], region [])
       ]);
  rejects_body "E_DIVERGENT_COLLECTIVE"
    (split Lt wide @ [ If (None, wide, region [ Barrier Execution.Cta ], region []) ]);
  (* tid = 256 elects one lane, which is divergent inside its warp. *)
  rejects_body "E_DIVERGENT_COLLECTIVE"
    (split Eq elect
     @ [ Const_f32 (lane_value, 1.)
       ; If (None, elect, region [ Warp_sum_f32 (reduced, lane_value) ], region [])
       ]);
  (* tid + 1 < 256 is warp-aligned in size but not in phase. *)
  rejects_body "E_DIVERGENT_COLLECTIVE"
    [ Thread_idx_x index
    ; Const_i32 (one, 1)
    ; Const_i32 (wide_limit, 256)
    ; Add_i32 (shifted, index, one)
    ; Const_f32 (lane_value, 1.)
    ; Compare (shifted_cond, Lt, shifted, wide_limit)
    ; If (None, shifted_cond, region [ Warp_sum_f32 (reduced, lane_value) ], region [])
    ];
  (* The same analysis picks bra.uni over a divergent branch in PTX. *)
  let branch_ptx comparison result =
    let one = fresh I32 in
    ptx_of
      (split comparison result
       @ [ If (None, result, region [ Const_i32 (one, 1) ], region []) ])
  in
  let uniform what comparison result =
    let emitted = branch_ptx comparison result in
    if not (contains emitted "bra.uni") || contains emitted "bra L" then
      failwith (what ^ " must lower to bra.uni")
  in
  uniform "tid < 256" Lt wide;
  uniform "tid >= 256" Ge (fresh Bool);
  let elect_ptx = branch_ptx Eq elect in
  if contains elect_ptx "bra.uni" || not (contains elect_ptx "bra L") then
    failwith "tid = 256 must stay divergent"

let uniform_loop_compare () =
  (* A loop induction variable is uniform, so comparing it is too. *)
  let trip_count = fresh I32 in
  let induction = fresh I32 and zero = fresh I32 and one = fresh I32 in
  let at_zero = fresh Bool in
  let emitted =
    Compiler.compile_ptx
      { branch_kernel with
        args = branch_kernel.args @ [ { name = "k"; value = trip_count } ]
      ; body =
          [ Const_i32 (zero, 0)
          ; Add_i32 (induction, zero, zero)
          ; For (induction, trip_count, 1,
                 [ Compare (at_zero, Eq, induction, zero)
                 ; If (None, at_zero, region [ Const_i32 (one, 1) ], region [])
                 ])
          ]
      }
  in
  if contains emitted "bra L" then
    failwith "a uniform loop induction compare must be bra.uni"

(* --- driver ----------------------------------------------------------- *)

let () =
  if Array.length Sys.argv = 2 && Sys.argv.(1) = "--ptx" then begin
    ok branch_kernel;
    print_string (Compiler.compile_ptx branch_kernel)
  end
  else begin
    List.iter
      (fun (name, test) ->
        try test () with
        | Failure message -> failwith (name ^ ": " ^ message)
        | Verifier.Invalid_kernel errors ->
            failwith
              (name ^ ": unexpected "
               ^ String.concat ","
                   (List.map (fun e -> e.Verifier.code) errors)))
      [ "branch lowers to a join", branch_lowers_to_a_join
      ; "branch scope is enforced", branch_scope_is_enforced
      ; "collectives need uniform control", collectives_need_uniform_control
      ; "remap is collective", remap_is_collective
      ; "shared tiles join at barriers", shared_tiles_join_at_barriers
      ; "branch store invalidates the load cache",
        branch_store_invalidates_the_load_cache
      ; "inherited pointers stay distinct", inherited_pointers_stay_distinct
      ; "warp aligned splits", warp_aligned_splits
      ; "uniform loop compare", uniform_loop_compare
      ];
    print_endline
      "structured branches, scope, uniformity, shared joins, and optimizer \
       checks passed"
  end
