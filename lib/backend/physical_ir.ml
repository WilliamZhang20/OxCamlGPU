(* Physicalization expands one logical tensor SSA value into the scalar values
   held by each lane. Supports 1D and rank-2 f32 tiles on one 32-lane warp. *)
type register = { index : int; dtype : Gpu_type.dtype option; value : Ir.value }
type value = {
  semantic : Ir.value;
  registers : register list;
  layout_mapping : Layout.mapped_register option;
}
type kernel = { target : Ptx_ir.kernel; values : (int * value) list }

type tile_slot_address =
  | Linear of { base : int; lane_stride : int }
  | Rank2 of {
      tile_cols : int;
      row_offset : int;
      col_offset : int;
      row_stride : int;
    }

let mapping_for_layout = function
  | Some (Layout.Mapped_register mapping) -> Some mapping
  | Some (Layout.Register distribution) -> Some {
      Layout.elements_per_lane=distribution.elements_per_lane;
      lanes_per_subgroup=distribution.lanes_per_subgroup;
      subgroups_per_cta=distribution.subgroups_per_cta;
      lane_mapping=Layout.Blocked;
      lane_order=distribution.order;
      register_order=distribution.order;
      subgroup_order=distribution.order }
  | _ -> None

let max_value_id target =
  List.fold_left (fun n v -> max n v.Ir.id) (-1) (Ptx_ir.all_values target)

let register_elements = function
  | Layout.Register r -> r.Layout.elements_per_lane
  | Layout.Mapped_register r -> r.Layout.elements_per_lane
  | _ -> invalid_arg "register layout expected"

let register_lanes = function
  | Layout.Register r -> r.Layout.lanes_per_subgroup
  | Layout.Mapped_register r -> r.Layout.lanes_per_subgroup
  | _ -> invalid_arg "register layout expected"

let register_subgroups = function
  | Layout.Register r -> r.Layout.subgroups_per_cta
  | Layout.Mapped_register r -> r.Layout.subgroups_per_cta
  | _ -> invalid_arg "register layout expected"

let tensor_register_count value = match value.Ir.ty, value.Ir.layout with
  | Gpu_type.Tensor (shape, _), Some layout ->
      let elements=List.fold_left ( * ) 1 (register_elements layout)
      and lanes=List.fold_left ( * ) 1 (register_lanes layout)
      and subgroups=List.fold_left ( * ) 1 (register_subgroups layout) in
      if lanes <> 32 || subgroups <> 1 then
        invalid_arg "physicalization currently requires one 32-lane subgroup per CTA";
      ignore (Layout.static_extents shape);
      elements
  | Gpu_type.Tensor _, _ -> invalid_arg "physicalization requires a tensor with a register layout"
  | _ -> 1

let iter_coordinates extents f =
  let rank = List.length extents in
  let rec loop coords dim =
    if dim=rank then f (List.rev coords)
    else
      let extent=List.nth extents dim in
      for i=0 to extent-1 do loop (i::coords) (dim+1) done
  in
  loop [] 0

let tile_slot_addresses mapping extents strides =
  let elements=List.fold_left ( * ) 1 mapping.Layout.elements_per_lane in
  let lanes=List.fold_left ( * ) 1 mapping.Layout.lanes_per_subgroup in
  let subgroups=List.fold_left ( * ) 1 mapping.Layout.subgroups_per_cta in
  if lanes <> 32 || subgroups <> 1 then
    invalid_arg "physicalization currently requires one 32-lane subgroup per CTA";
  match extents, strides with
  | [extent], [stride] ->
      let offsets=Array.make (32 * elements) (-1) in
      for logical=0 to extent-1 do
        let hardware = Layout.mapped_to_hardware mapping [logical] in
        if hardware.cta <> [0] || hardware.subgroup_id <> 0 ||
           hardware.lane_id < 0 || hardware.lane_id >= 32 ||
           hardware.register_id < 0 || hardware.register_id >= elements then
          invalid_arg "physicalization received an unsupported tile mapping";
        let slot=hardware.lane_id * elements + hardware.register_id in
        if offsets.(slot) <> -1 then
          invalid_arg "physicalization layout maps multiple elements to one lane register";
        offsets.(slot) <- logical * stride
      done;
      List.init elements (fun index ->
        let base=offsets.(index) and next=offsets.(elements + index) in
        let lane_stride=next-base in
        for lane=0 to 31 do
          if offsets.(lane * elements + index) <> base + lane * lane_stride then
            invalid_arg "physicalization: layout is not affine in lane and register coordinates"
        done;
        Linear { base; lane_stride })
  | [_; cols], [row_stride; col_stride] ->
      if col_stride <> 1 then
        invalid_arg "physicalization currently requires unit stride on the fastest MemRef dimension";
      let elements=List.fold_left ( * ) 1 mapping.Layout.elements_per_lane in
      let offsets=Array.make (32 * elements) (-1) in
      iter_coordinates extents (fun logical ->
        let hardware = Layout.mapped_to_hardware mapping logical in
        if hardware.subgroup_id <> 0 ||
           hardware.lane_id < 0 || hardware.lane_id >= 32 ||
           hardware.register_id < 0 || hardware.register_id >= elements then
          invalid_arg "physicalization received an unsupported tile mapping";
        let slot=hardware.lane_id * elements + hardware.register_id in
        if offsets.(slot) <> -1 then
          invalid_arg "physicalization layout maps multiple elements to one lane register";
        offsets.(slot) <- Layout.linear_offset strides logical);
      let row_major_rank2 () =
        if elements <> 1 then raise Exit;
        let expected_lane logical =
          match logical with [r; c] -> r * cols + c | _ -> assert false in
        iter_coordinates extents (fun logical ->
          let hardware = Layout.mapped_to_hardware mapping logical in
          if hardware.lane_id <> expected_lane logical then raise Exit);
        [Rank2 { tile_cols=cols; row_offset=0; col_offset=0; row_stride }] in
      let linear_affine () =
        List.init elements (fun index ->
          let base=offsets.(index) and next=offsets.(elements + index) in
          let lane_stride=next-base in
          for lane=0 to 31 do
            if offsets.(lane * elements + index) <> base + lane * lane_stride then
              raise Exit
          done;
          Linear { base; lane_stride }) in
      (try row_major_rank2 () with Exit ->
         try linear_affine () with Exit ->
           invalid_arg "rank-2 physicalization requires row-major lane ownership or affine linear addressing")
  | _ -> invalid_arg "physicalization currently supports rank-1 or rank-2 tiles only"

let scalar_dtype value = match value.Ir.ty with
  | Gpu_type.F32 -> Some Gpu_type.Float32
  | Gpu_type.I32 | Gpu_type.Bool -> Some Gpu_type.Int32
  | _ -> None

let physicalize target =
  let next_id = ref (max_value_id target + 1) in
  let table = Hashtbl.create 32 in
  let values=Ptx_ir.all_values target in
  List.iter (fun semantic -> if not (Hashtbl.mem table semantic.Ir.id) then begin
    let registers=match semantic.Ir.ty with
      | Gpu_type.Tensor (_, dtype) ->
          let count=tensor_register_count semantic in
          List.init count (fun index ->
            let scalar_ty=match dtype with
              | Gpu_type.Float32 -> Gpu_type.F32 | Gpu_type.Int32 -> Gpu_type.I32 in
            let scalar=if index=0 then { semantic with Ir.ty=scalar_ty; layout=None }
              else begin
                let id= !next_id in incr next_id;
                Ir.make_value id scalar_ty
              end in
            { index; dtype=Some dtype; value=scalar })
      | _ -> [{ index=0; dtype=scalar_dtype semantic; value=semantic }] in
    Hashtbl.add table semantic.Ir.id { semantic; registers;
      layout_mapping=mapping_for_layout semantic.Ir.layout }
  end) values;
  let physical id = Hashtbl.find table id in
  let slot value index =
    match List.find_opt (fun register -> register.index=index) (physical value.Ir.id).registers with
    | Some register -> register.value
    | None -> invalid_arg (Printf.sprintf "physicalization: tensor %d has no register slot %d" value.Ir.id index) in
  let scalar value =
    let p=physical value.Ir.id in
    match p.registers with
    | [register] -> register.value
    | _ -> invalid_arg (Printf.sprintf "physicalization: tensor %d used as a scalar" value.Ir.id) in
  let addresses tensor memref =
    let p=physical tensor.Ir.id in
    let mapping=match p.layout_mapping with
      | Some mapping -> mapping
      | None -> invalid_arg "physicalization: tensor has no layout mapping" in
    let shape=match tensor.Ir.ty with
      | Gpu_type.Tensor (shape, _) -> shape
      | _ -> invalid_arg "tile addressing requires a tensor" in
    let extents = Layout.static_extents shape in
    let strides=match memref.Ir.ty with
      | Gpu_type.MemRef (mem_shape, _, _) ->
          if mem_shape <> shape then
            invalid_arg "tile MemRef shape must match the tensor shape";
          Layout.memref_strides mem_shape memref.Ir.layout
      | _ -> invalid_arg "tile addressing requires a MemRef" in
    tile_slot_addresses mapping extents strides in
  let lower_operation operation = match operation with
    | Ptx_ir.Tile_load_f32 (tensor, memref) ->
        List.mapi (fun index addr -> match addr with
          | Linear { base; lane_stride } ->
              Ptx_ir.Tile_load_f32_indexed (slot tensor index, scalar memref, base, lane_stride)
          | Rank2 { tile_cols; row_offset; col_offset; row_stride } ->
              Ptx_ir.Tile_load_f32_2d (slot tensor index, scalar memref,
                tile_cols, row_offset, col_offset, row_stride, None))
          (addresses tensor memref)
    | Ptx_ir.Tile_store_f32 (memref, tensor) ->
        List.mapi (fun index addr -> match addr with
          | Linear { base; lane_stride } ->
              Ptx_ir.Tile_store_f32_indexed (scalar memref, slot tensor index, base, lane_stride)
          | Rank2 { tile_cols; row_offset; col_offset; row_stride } ->
              Ptx_ir.Tile_store_f32_2d (scalar memref, slot tensor index,
                tile_cols, row_offset, col_offset, row_stride, None))
          (addresses tensor memref)
    | Ptx_ir.Tile_load_f32_masked (tensor, memref, rows, cols) ->
        List.mapi (fun index addr -> match addr with
          | Rank2 { tile_cols; row_offset; col_offset; row_stride } ->
              Ptx_ir.Tile_load_f32_2d (slot tensor index, scalar memref,
                tile_cols, row_offset, col_offset, row_stride, Some (scalar rows, scalar cols))
          | Linear _ -> invalid_arg "masked tile loads currently require rank-2 tiles")
          (addresses tensor memref)
    | Ptx_ir.Tile_store_f32_masked (memref, tensor, rows, cols) ->
        List.mapi (fun index addr -> match addr with
          | Rank2 { tile_cols; row_offset; col_offset; row_stride } ->
              Ptx_ir.Tile_store_f32_2d (scalar memref, slot tensor index,
                tile_cols, row_offset, col_offset, row_stride, Some (scalar rows, scalar cols))
          | Linear _ -> invalid_arg "masked tile stores currently require rank-2 tiles")
          (addresses tensor memref)
    | Ptx_ir.Tensor_scale_f32 (dst, src, factor) ->
        List.init (List.length (physical dst.Ir.id).registers) (fun index ->
          Ptx_ir.Mul_f32 (slot dst index, slot src index, scalar factor))
    | Ptx_ir.Tensor_mul_f32 (dst, left, right) ->
        List.init (List.length (physical dst.Ir.id).registers) (fun index ->
          Ptx_ir.Mul_f32 (slot dst index, slot left index, slot right index))
    | Ptx_ir.Tensor_lane_f32 (dst, tensor) ->
        (match List.init (List.length (physical tensor.Ir.id).registers) (slot tensor) with
         | [] -> invalid_arg "physicalization: tensor has no registers"
         | first :: rest ->
             [Ptx_ir.Move (scalar dst, first)] @
             List.map (fun register -> Ptx_ir.Add_f32 (scalar dst, scalar dst, register)) rest)
    | Ptx_ir.Const_bool (dst, b) -> [Ptx_ir.Const_bool (scalar dst, b)]
    | Ptx_ir.Sub_i32 (dst,a,b) -> [Ptx_ir.Sub_i32 (scalar dst,scalar a,scalar b)]
    | Ptx_ir.Mul_i32 (dst,a,b) -> [Ptx_ir.Mul_i32 (scalar dst,scalar a,scalar b)]
    | Ptx_ir.Div_i32 (dst,a,b) -> [Ptx_ir.Div_i32 (scalar dst,scalar a,scalar b)]
    | Ptx_ir.Min_i32 (dst,a,b) -> [Ptx_ir.Min_i32 (scalar dst,scalar a,scalar b)]
    | Ptx_ir.Rem_i32 (dst,a,b) -> [Ptx_ir.Rem_i32 (scalar dst,scalar a,scalar b)]
    | Ptx_ir.Compare (dst,c,a,b) -> [Ptx_ir.Compare (scalar dst,c,scalar a,scalar b)]
    | Ptx_ir.Move (dst,src) -> [Ptx_ir.Move (scalar dst,scalar src)]
    | Ptx_ir.Const_i32 (dst,n) -> [Ptx_ir.Const_i32 (scalar dst,n)]
    | Ptx_ir.Const_f32 (dst,n) -> [Ptx_ir.Const_f32 (scalar dst,n)]
    | Ptx_ir.Add_i32 (dst,a,b) -> [Ptx_ir.Add_i32 (scalar dst,scalar a,scalar b)]
    | Ptx_ir.Add_f32 (dst,a,b) -> [Ptx_ir.Add_f32 (scalar dst,scalar a,scalar b)]
    | Ptx_ir.Mul_f32 (dst,a,b) -> [Ptx_ir.Mul_f32 (scalar dst,scalar a,scalar b)]
    | Ptx_ir.Mad_f32 (dst,a,b,c) -> [Ptx_ir.Mad_f32 (scalar dst,scalar a,scalar b,scalar c)]
    | Ptx_ir.Thread_idx_x dst -> [Ptx_ir.Thread_idx_x (scalar dst)]
    | Ptx_ir.Global_idx_x dst -> [Ptx_ir.Global_idx_x (scalar dst)]
    | Ptx_ir.Block_idx_x dst -> [Ptx_ir.Block_idx_x (scalar dst)]
    | Ptx_ir.Block_idx_y dst -> [Ptx_ir.Block_idx_y (scalar dst)]
    | Ptx_ir.Gep_f32 (dst,p,index) -> [Ptx_ir.Gep_f32 (scalar dst,scalar p,scalar index)]
    | Ptx_ir.Load_f32 (dst,p) -> [Ptx_ir.Load_f32 (scalar dst,scalar p)]
    | Ptx_ir.Load_f32_masked (dst,p,index,bound) ->
        [Ptx_ir.Load_f32_masked (scalar dst,scalar p,scalar index,scalar bound)]
    | Ptx_ir.Load_f32x4 (a,b,c,d,p) ->
        [Ptx_ir.Load_f32x4 (scalar a,scalar b,scalar c,scalar d,scalar p)]
    | Ptx_ir.Shared_load_f32 (dst,p,index) ->
        [Ptx_ir.Shared_load_f32 (scalar dst,scalar p,scalar index)]
    | Ptx_ir.Shared_load_f32x4 (a,b,c,d,p,index) ->
        [Ptx_ir.Shared_load_f32x4 (scalar a,scalar b,scalar c,scalar d,scalar p,scalar index)]
    | Ptx_ir.Shared_store_f32 (p,value,index) ->
        [Ptx_ir.Shared_store_f32 (scalar p,scalar value,scalar index)]
    | Ptx_ir.Shared_store_f32x4 (p,a,b,c,d,index) ->
        [Ptx_ir.Shared_store_f32x4 (scalar p,scalar a,scalar b,scalar c,scalar d,scalar index)]
    | Ptx_ir.Cp_async_shared_f32x4 (p,index,g) ->
        [Ptx_ir.Cp_async_shared_f32x4 (scalar p,scalar index,scalar g)]
    | Ptx_ir.Cp_async_commit -> [Ptx_ir.Cp_async_commit]
    | Ptx_ir.Cp_async_wait_all -> [Ptx_ir.Cp_async_wait_all]
    | Ptx_ir.Mbarrier_init (m,c,p) ->
        [Ptx_ir.Mbarrier_init (scalar m, scalar c, Option.map scalar p)]
    | Ptx_ir.Mbarrier_arrive_expect_tx (m,t,p) ->
        [Ptx_ir.Mbarrier_arrive_expect_tx (scalar m, scalar t, Option.map scalar p)]
    | Ptx_ir.Mbarrier_arrive (m,p) ->
        [Ptx_ir.Mbarrier_arrive (scalar m, Option.map scalar p)]
    | Ptx_ir.Mbarrier_try_wait_parity (m,p) ->
        [Ptx_ir.Mbarrier_try_wait_parity (scalar m, scalar p)]
    | Ptx_ir.Mbarrier_slot (d,set,ix) ->
        [Ptx_ir.Mbarrier_slot (d, scalar set, scalar ix)]
    | Ptx_ir.Tma_load_2d (s,i,t,c0,c1,m,p) ->
        [Ptx_ir.Tma_load_2d
           ( scalar s, scalar i, scalar t, scalar c0, scalar c1, scalar m
           , Option.map scalar p )]
    | Ptx_ir.Fence_proxy_async -> [Ptx_ir.Fence_proxy_async]
    | Ptx_ir.Gmma_descriptor (d, s, i, fields) ->
        [Ptx_ir.Gmma_descriptor (scalar d, scalar s, scalar i, fields)]
    | Ptx_ir.Wgmma_fence -> [Ptx_ir.Wgmma_fence]
    | Ptx_ir.Wgmma_mma_tf32 (acc,da,db,n,scale) ->
        [Ptx_ir.Wgmma_mma_tf32
           (List.map scalar acc, scalar da, scalar db, n, scale)]
    | Ptx_ir.Wgmma_commit_group -> [Ptx_ir.Wgmma_commit_group]
    | Ptx_ir.Wgmma_wait_group n -> [Ptx_ir.Wgmma_wait_group n]
    | Ptx_ir.Store_f32 (p,value) -> [Ptx_ir.Store_f32 (scalar p,scalar value)]
    | Ptx_ir.Store_f32x2 (p,a,b) ->
        [Ptx_ir.Store_f32x2 (scalar p,scalar a,scalar b)]
    | Ptx_ir.Store_f32x4 (p,a,b,c,d) ->
        [Ptx_ir.Store_f32x4 (scalar p,scalar a,scalar b,scalar c,scalar d)]
    | Ptx_ir.Store_f32_masked (p,value,index,bound) ->
        [Ptx_ir.Store_f32_masked (scalar p,scalar value,scalar index,scalar bound)]
    | Ptx_ir.Tile_load_f32_indexed (dst,p,base,stride) ->
        [Ptx_ir.Tile_load_f32_indexed (scalar dst,scalar p,base,stride)]
    | Ptx_ir.Tile_store_f32_indexed (p,value,base,stride) ->
        [Ptx_ir.Tile_store_f32_indexed (scalar p,scalar value,base,stride)]
    | Ptx_ir.Tile_load_f32_2d (dst,p,cols,ro,co,rs,bounds) ->
        [Ptx_ir.Tile_load_f32_2d (scalar dst,scalar p,cols,ro,co,rs,
          Option.map (fun (a,b) -> scalar a, scalar b) bounds)]
    | Ptx_ir.Tile_store_f32_2d (p,value,cols,ro,co,rs,bounds) ->
        [Ptx_ir.Tile_store_f32_2d (scalar p,scalar value,cols,ro,co,rs,
          Option.map (fun (a,b) -> scalar a, scalar b) bounds)]
    | Ptx_ir.Store_grid_leader_f32 (p,value) ->
        [Ptx_ir.Store_grid_leader_f32 (scalar p,scalar value)]
    | Ptx_ir.Barrier_cta | Ptx_ir.Label _ | Ptx_ir.Jump _ | Ptx_ir.Jump_uni _
    | Ptx_ir.Branch _ | Ptx_ir.Branch_uni _
    | Ptx_ir.Shared_alloc _
    | Ptx_ir.Warp_butterfly_sum_f32 _ | Ptx_ir.Return as op -> [op]
  in
  let body=List.concat_map lower_operation target.Ptx_ir.body in
  let target={target with Ptx_ir.body=body} in
  { target; values=Hashtbl.fold (fun id value acc -> (id,value)::acc) table [] }

let lookup kernel value = List.assoc_opt value.Ir.id kernel.values

let coordinate kernel value logical = match lookup kernel value with
  | Some { layout_mapping=Some mapping; _ } -> Layout.mapped_to_hardware mapping logical
  | Some _ -> invalid_arg "physical value has no tensor coordinate mapping"
  | None -> invalid_arg "value has no physical representation"
