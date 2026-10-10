(* Versioned frontend AST transported through [.gpu] metadata. Public by
   construction: the adapter writes it and the frontend reads it. *)
open Gpu_type
open Gpu_mode

type slot = {
  loc : Source_span.t;
  ty : ty;
  ownership : ownership;
  locality : locality;
  domain_portability : domain_portability;
  (* Always Boundary_unspecified on the wire; Kernel_frontend.lower derives
     Boundary_portable for buffer/pointer formals. *)
  gpu_boundary : gpu_boundary;
  permission : permission;
}

type argument = {
  index : int;
  slot : slot;
}

type atom =
  | Arg of int
  | Value of int

type memory =
  | Buffer_arg of int
  | Pointer of atom

(* Source-frontend operations. Every constructor must encode/parse in
   Gpu_metadata and lower in Kernel_frontend. Authors compose hardware ops
   explicitly — there is no whole-kernel matmul expand. *)
type operation =
  | Const_bool of int * bool
  | Const_i32 of int * int
  | Const_f32 of int * float
  | Add_i32 of int * atom * atom
  | Sub_i32 of int * atom * atom
  | Mul_i32 of int * atom * atom
  | Div_i32 of int * atom * atom
  | Rem_i32 of int * atom * atom
  | Compare of int * comparison * atom * atom
  | If of (int * ty) option * atom * region * region
  | Thread_idx_x of int
  | Global_idx_x of int
  | Block_idx_x of int
  | Block_idx_y of int
  | Gep_f32 of int * memory * atom
  | Load_f32 of int * atom
  | Load_f32_masked of int * atom * atom * atom
  | Load_f32x4 of int * int * int * int * atom
  | Store_f32 of atom * atom
  | Store_f32_masked of atom * atom * atom * atom
  | Store_f32x4 of atom * atom * atom * atom * atom
  | Add_f32 of int * atom * atom
  | Mul_f32 of int * atom * atom
  | Mad_f32 of int * atom * atom * atom
  | Warp_reduce_sum_f32 of int * atom
  | Store_grid_leader_f32 of memory * atom
  (* CTA shared: [dst] is MemRef(rows×cols,f32,Shared); extents are static. *)
  | Shared_alloc of int * int * int * [`Plain | `Wgmma]
  | Shared_load_f32 of int * atom * atom
  | Shared_load_f32x4 of int * int * int * int * atom * atom
  | Shared_store_f32 of atom * atom * atom
  | Shared_store_f32x4 of atom * atom * atom * atom * atom * atom
  | Barrier_cta
  | Cp_async_f32x4 of atom * atom * memory * atom
  | Cp_async_commit
  | Cp_async_wait_all
  (* Hopper / SM90 — author-scheduled; optional elect predicate. *)
  | Mbarrier_alloc of int
  (* A set of [count] mbarriers, and one of them at a dynamic index. *)
  | Mbarrier_set_alloc of int * int
  | Mbarrier_slot of int * atom * atom
  | Mbarrier_init of atom * atom * atom option
  | Mbarrier_arrive_expect_tx of atom * atom * atom option
  | Mbarrier_arrive of atom * atom option
  | Mbarrier_try_wait_parity of atom * atom
  | Tma_load_2d of atom * atom * atom * atom * atom * atom * atom option
  | Fence_proxy_async
  | Wgmma_fence
  | Gmma_descriptor of int * atom * atom
  | Wgmma_mma_tf32 of int list * atom * atom * int * bool
  | Wgmma_commit_group
  | Wgmma_wait_group of int
  (** Inclusive OCaml [for] with step 1 is lowered to exclusive [Ir.For]
      (limit = hi+1). Application IR may also emit exclusive limits directly. *)
  | For of int * atom * atom * int * region

and instruction = {
  op : operation;
  loc : Source_span.t;
}

and region = {
  body : instruction list;
  yield : atom option;
}

type t = {
  name : string;
  args : argument list;
  result : slot;
  body : instruction list;
  threads : int option;
}

let instruction ?(loc = Source_span.synthetic) op = { op; loc }
