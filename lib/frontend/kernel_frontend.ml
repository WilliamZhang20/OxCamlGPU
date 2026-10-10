open Gpu_type
open Gpu_mode
open Ir

(* Buffer formals crossed the host/device launch boundary. That GPU-boundary
   fact is derived here from the formal's type; the OxCaml adapter does not
   import it (wire slots stay Boundary_unspecified). *)
let entry_gpu_boundary = function
  | Ptr _ | MemRef _ | Tensor_map -> Boundary_portable
  | _ -> Boundary_unspecified

let resolve_atom ~loc ~args definitions = function
  | Kernel_ast.Arg i ->
      if i < 0 || i >= List.length args then
        Source_span.fail loc "invalid argument reference"
      else (List.nth args i).value
  | Kernel_ast.Value id ->
      match Hashtbl.find_opt definitions id with
      | Some v -> v
      | None -> Source_span.fail loc "undefined or out-of-scope frontend value"

(* Every Kernel_ast.operation constructor has a lowering arm below. *)
(* The packed GMMA descriptor fields are a function of the shared tile a WGMMA
   operand lives in, so an author names the tile and the compiler fills them
   in. A 128-byte swizzle makes the atom eight rows tall, so consecutive atoms
   sit 8 row-bytes apart, counted in the descriptor's 16-byte units; a K-major
   operand's leading offset is one such unit. Only the swizzle and row size
   WGMMA actually accepts are derivable, and anything else is rejected rather
   than guessed. *)
let gmma_fields_of_tile loc (tile : Ir.value) =
  match tile.Ir.ty, tile.Ir.layout with
  | Gpu_type.MemRef ([Gpu_type.Static _; Gpu_type.Static cols], Gpu_type.Float32,
                     Gpu_type.Shared),
    Some (Layout.Shared { order = [1; 0]; swizzle = Layout.Swizzle_128B; _ })
    when cols * 4 = 128 ->
      { Ir.leading = 1; stride = 8 * (cols * 4) / 16; layout_type = 1 }
  | Gpu_type.MemRef ([Gpu_type.Static _; Gpu_type.Static cols], Gpu_type.Float32,
                     Gpu_type.Shared),
    Some (Layout.Shared { order = [1; 0]; swizzle = Layout.Swizzle_128B; _ }) ->
      Source_span.fail loc
        (Printf.sprintf
           "a 128-byte-swizzled WGMMA tile needs 32 f32 columns, not %d" cols)
  | _ ->
      Source_span.fail loc
        "a WGMMA operand descriptor needs a K-major swizzled shared tile \
         from Gpu.shared_wgmma"

let lower (source : Kernel_ast.t) =
  if source.result.ty <> Unit then
    invalid_arg "GPU kernel entry points must return unit";

  let args =
    List.mapi
      (fun index (arg : Kernel_ast.argument) ->
        if arg.index <> index then invalid_arg "noncontiguous argument ids";
        let s = arg.slot in
        (* Imported OxCaml modes ride on [s]; wire gpu_boundary is ignored. *)
        let provenance =
          match s.ty with
          | Ptr _ | MemRef _ -> Some index
          | _ -> None
        in
        { name = "arg" ^ string_of_int index
        ; value =
            make_value
              ~loc:s.loc
              ~ownership:s.ownership
              ~locality:s.locality
              ~domain_portability:s.domain_portability
              ~gpu_boundary:(entry_gpu_boundary s.ty)
              ~permission:s.permission
              ~provenance
              index
              s.ty
        })
      source.args
  in

  let used = Hashtbl.create 32 in
  let next_temp = ref 0 in

  let rec lower_body definitions body =
    let atom loc = resolve_atom ~loc ~args definitions in

    List.concat_map
      (fun ({ op; loc } : Kernel_ast.instruction) ->
        let define
            ?(ownership = Aliased)
            ?(locality = Local)
            ?(domain_portability = Domain_portability_unspecified)
            ?(gpu_boundary = Boundary_unspecified)
            ?(permission = Read_only)
            ?provenance
            ?layout
            ?base
            id
            ty =
          (* Frontend SSA ids map to IR ids [List.length args + id]. *)
          if id < 0 || id > max_int - List.length args || Hashtbl.mem used id
          then Source_span.fail loc "invalid or duplicate frontend SSA id";
          Hashtbl.add used id ();
          let value =
            match base with
            | None ->
                make_value
                  ~loc
                  ~ownership
                  ~locality
                  ~domain_portability
                  ~gpu_boundary
                  ~permission
                  ?provenance
                  ?layout
                  (List.length args + id)
                  ty
            | Some base ->
                make_value
                  ~loc
                  ~ownership:Aliased
                  ~locality:Local
                  ~domain_portability:base.domain_portability
                  ~gpu_boundary:base.gpu_boundary
                  ~permission:base.permission
                  ~provenance:base.provenance
                  ?layout:base.layout
                  (List.length args + id)
                  ty
          in
          Hashtbl.add definitions id value;
          value
        in

        let fresh_temp ?(permission = Read_only) ?layout ty =
          let n = !next_temp in
          incr next_temp;
          make_value
            ~loc
            ~ownership:Aliased
            ~locality:Local
            ~permission
            ?layout
            (1_000_000_000 + n)
            ty
        in

        let shared_plain =
          Layout.Shared
            { vector_width = 1; order = [1; 0]; swizzle = Layout.No_swizzle }
        in
        let mbar_layout =
          Layout.Shared
            { vector_width = 1; order = [0]; swizzle = Layout.No_swizzle }
        in

        let memory = function
          | Kernel_ast.Buffer_arg i -> atom loc (Kernel_ast.Arg i)
          | Kernel_ast.Pointer p -> atom loc p
        in

        let opt_atom = Option.map (atom loc) in

        match op with
        | Kernel_ast.Const_i32 (d, n) -> [Const_i32 (define d I32, n)]
        | Kernel_ast.Const_f32 (d, n) -> [Const_f32 (define d F32, n)]
        | Kernel_ast.Const_bool (d, n) -> [Const_bool (define d Bool, n)]

        | Kernel_ast.Thread_idx_x d -> [Thread_idx_x (define d I32)]
        | Kernel_ast.Global_idx_x d -> [Global_idx_x (define d I32)]
        | Kernel_ast.Block_idx_x d -> [Block_idx_x (define d I32)]
        | Kernel_ast.Block_idx_y d -> [Block_idx_y (define d I32)]

        | Kernel_ast.Gep_f32 (d, p, i) ->
            let p = memory p and i = atom loc i in
            [Gep_f32 (define ~base:p d (Ptr (F32, Global)), p, i)]

        | Kernel_ast.Load_f32 (d, p) ->
            [Load_f32 (define d F32, atom loc p)]

        | Kernel_ast.Load_f32_masked (d, p, i, n) ->
            [Load_f32_masked
               (define d F32, atom loc p, atom loc i, atom loc n)]

        | Kernel_ast.Load_f32x4 (a, b, c, d, p) ->
            [Load_f32x4
               ( define a F32, define b F32, define c F32, define d F32
               , atom loc p )]

        | Kernel_ast.Add_f32 (d, a, b) ->
            [Add_f32 (define d F32, atom loc a, atom loc b)]
        | Kernel_ast.Mul_f32 (d, a, b) ->
            [Mul_f32 (define d F32, atom loc a, atom loc b)]
        | Kernel_ast.Mad_f32 (d, acc, a, b) ->
            (* dst = a * b + acc *)
            [Mad_f32 (define d F32, atom loc a, atom loc b, atom loc acc)]

        | Kernel_ast.Add_i32 (d, a, b) ->
            [Add_i32 (define d I32, atom loc a, atom loc b)]
        | Kernel_ast.Sub_i32 (d, a, b) ->
            [Sub_i32 (define d I32, atom loc a, atom loc b)]
        | Kernel_ast.Mul_i32 (d, a, b) ->
            [Mul_i32 (define d I32, atom loc a, atom loc b)]
        | Kernel_ast.Div_i32 (d, a, b) ->
            [Div_i32 (define d I32, atom loc a, atom loc b)]
        | Kernel_ast.Rem_i32 (d, a, b) ->
            [Rem_i32 (define d I32, atom loc a, atom loc b)]

        | Kernel_ast.Compare (d, c, a, b) ->
            [Compare (define d Bool, c, atom loc a, atom loc b)]

        | Kernel_ast.Warp_reduce_sum_f32 (d, v) ->
            [Warp_sum_f32 (define d F32, atom loc v)]

        | Kernel_ast.Store_f32 (p, v) ->
            [Store_f32 (atom loc p, atom loc v)]
        | Kernel_ast.Store_f32_masked (p, v, i, n) ->
            [Store_f32_masked
               (atom loc p, atom loc v, atom loc i, atom loc n)]
        | Kernel_ast.Store_f32x4 (p, a, b, c, d) ->
            [Store_f32x4
               ( atom loc p, atom loc a, atom loc b, atom loc c, atom loc d )]
        | Kernel_ast.Store_grid_leader_f32 (p, v) ->
            [Store_f32_grid_leader (memory p, atom loc v)]

        | Kernel_ast.Shared_alloc (d, rows, cols, layout_kind) ->
            if rows <= 0 || cols <= 0 then
              Source_span.fail loc "shared extents must be positive";
            let ty = MemRef ([Static rows; Static cols], Float32, Shared) in
            let layout =
              match layout_kind with
              | `Plain -> shared_plain
              | `Wgmma ->
                  Layout.Shared
                    { vector_width = 1
                    ; order = [1; 0]
                    ; swizzle = Layout.Swizzle_128B
                    }
            in
            let m =
              define ~permission:Read_write ~layout d ty
            in
            [Shared_alloc m]

        | Kernel_ast.Shared_load_f32 (d, m, i) ->
            [Shared_load_f32 (define d F32, atom loc m, atom loc i)]
        | Kernel_ast.Shared_load_f32x4 (a, b, c, d, m, i) ->
            [Shared_load_f32x4
               ( define a F32, define b F32, define c F32, define d F32
               , atom loc m, atom loc i )]
        | Kernel_ast.Shared_store_f32 (m, v, i) ->
            [Shared_store_f32 (atom loc m, atom loc v, atom loc i)]
        | Kernel_ast.Shared_store_f32x4 (m, a, b, c, d, i) ->
            [Shared_store_f32x4
               ( atom loc m, atom loc a, atom loc b, atom loc c, atom loc d
               , atom loc i )]

        | Kernel_ast.Barrier_cta -> [Barrier Execution.Cta]

        | Kernel_ast.Cp_async_f32x4 (m, i, g, gi) ->
            let g = memory g and gi = atom loc gi in
            let ptr = fresh_temp (Ptr (F32, Global)) in
            [ Gep_f32 (ptr, g, gi)
            ; Cp_async_shared_f32x4 (atom loc m, atom loc i, ptr)
            ]
        | Kernel_ast.Cp_async_commit -> [Cp_async_commit]
        | Kernel_ast.Cp_async_wait_all -> [Cp_async_wait_all]

        | Kernel_ast.Mbarrier_alloc d ->
            let ty = MemRef ([Static 2], Int32, Shared) in
            let m =
              define ~permission:Read_write ~layout:mbar_layout d ty
            in
            [Shared_alloc m]
        | Kernel_ast.Mbarrier_set_alloc (d, count) ->
            if count <= 0 then
              Source_span.fail loc "an mbarrier set needs a positive count";
            (* Each mbarrier is two i32 words. *)
            let ty = MemRef ([Static (2 * count)], Int32, Shared) in
            let m = define ~permission:Read_write ~layout:mbar_layout d ty in
            [Shared_alloc m]
        | Kernel_ast.Mbarrier_slot (d, set, index) ->
            let ty = MemRef ([Static 2], Int32, Shared) in
            let dst = define ~permission:Read_write ~layout:mbar_layout d ty in
            [Mbarrier_slot (dst, atom loc set, atom loc index)]
        | Kernel_ast.Mbarrier_init (m, c, p) ->
            [Mbarrier_init (atom loc m, atom loc c, opt_atom p)]
        | Kernel_ast.Mbarrier_arrive_expect_tx (m, tx, p) ->
            [Mbarrier_arrive_expect_tx (atom loc m, atom loc tx, opt_atom p)]
        | Kernel_ast.Mbarrier_arrive (m, p) ->
            [Mbarrier_arrive (atom loc m, opt_atom p)]
        | Kernel_ast.Mbarrier_try_wait_parity (m, parity) ->
            [Mbarrier_try_wait_parity (atom loc m, atom loc parity)]

        | Kernel_ast.Tma_load_2d (sm, idx, tmap, c0, c1, mbar, p) ->
            [Tma_load_2d
               ( atom loc sm, atom loc idx, atom loc tmap, atom loc c0
               , atom loc c1, atom loc mbar, opt_atom p )]
        | Kernel_ast.Fence_proxy_async -> [Fence_proxy_async]
        | Kernel_ast.Wgmma_fence -> [Wgmma_fence]
        | Kernel_ast.Gmma_descriptor (d, sm, idx) ->
            let tile = atom loc sm in
            [Gmma_descriptor
               (define d U64, tile, atom loc idx,
                gmma_fields_of_tile loc tile)]
        | Kernel_ast.Wgmma_mma_tf32 (acc, da, db, n, scale) ->
            let acc =
              List.map
                (fun id ->
                  match Hashtbl.find_opt definitions id with
                  | Some v -> v
                  | None ->
                      Source_span.fail loc "undefined WGMMA accumulator")
                acc
            in
            [Wgmma_mma_tf32 (acc, atom loc da, atom loc db, n, scale)]
        | Kernel_ast.Wgmma_commit_group -> [Wgmma_commit_group]
        | Kernel_ast.Wgmma_wait_group n -> [Wgmma_wait_group n]

        | Kernel_ast.For (ind, lo, limit, step, body) ->
            if step <= 0 then
              Source_span.fail loc "loop step must be positive";
            let induction = define ind I32 in
            let lo = atom loc lo and limit = atom loc limit in
            let zero = fresh_temp I32 in
            let env = Hashtbl.copy definitions in
            Hashtbl.replace env ind induction;
            let loop_body = lower_body env body.body in
            [ Const_i32 (zero, 0)
            ; Add_i32 (induction, lo, zero)
            ; For (induction, limit, step, loop_body)
            ]

        | Kernel_ast.If (dst, c, a, b) ->
            let c = atom loc c in
            let region (r : Kernel_ast.region) =
              let env = Hashtbl.copy definitions in
              let body = lower_body env r.body in
              let yield =
                Option.map (fun v -> resolve_atom ~loc ~args env v) r.yield
              in
              { body; yield }
            in
            let a = region a and b = region b in
            let dst =
              Option.map
                (fun (d, t) ->
                  match a.yield, b.yield with
                  | Some left, Some right ->
                      define
                        ~ownership:Aliased
                        ~locality:(meet_locality left.locality right.locality)
                        ~domain_portability:
                          (meet_domain_portability
                             left.domain_portability
                             right.domain_portability)
                        ~gpu_boundary:
                          (meet_gpu_boundary
                             left.gpu_boundary
                             right.gpu_boundary)
                        ~permission:
                          (meet_permission left.permission right.permission)
                        d
                        t
                  | _ -> define d t)
                dst
            in
            [If (dst, c, a, b)])
      body
  in

  let body = lower_body (Hashtbl.create 32) source.body in
  let threads_per_cta =
    match source.threads with
    | None -> None
    | Some n when n >= 1 && n <= 1024 -> Some n
    | Some n ->
        Source_span.fail source.result.loc
          (Printf.sprintf "CTA thread count %d is outside 1..1024" n)
  in
  let kernel = { name = source.name; args; body; threads_per_cta } in
  Verifier.verify_exn kernel;
  kernel

let () =
  (* Inline formatting: [Source_span.to_string] is not [portable], but
     [Printexc.Safe.register_printer] requires a portable closure. *)
  Printexc.Safe.register_printer (function
    | Source_span.Error (loc, message) ->
        Some
          (Printf.sprintf "%s:%d:%d: %s" loc.file loc.start_line
             (loc.start_col + 1) message)
    | _ -> None)
