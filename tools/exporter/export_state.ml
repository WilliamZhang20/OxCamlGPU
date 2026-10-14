(* Vocabulary shared by every stage of the Typedtree exporter: the values an
   expression can evaluate to, the emission state it threads, the limits on
   compile-time evaluation, and the small coercions primitives are built from. *)
open Typedtree
module K = Kernel_ast
open Typedtree_support

(* What an expression denotes while the kernel is being traced. *)
type binding =
  | Scalar of K.atom * (int64 * int64) option
  | Const_bool of K.atom * bool
  | Memory of K.memory
  | Shared of K.atom
  | Mbarrier of K.atom
  | Tensor_map of K.atom
  | Wgmma_acc of int list
  | Func of func
  | Void

and func = {
  params : (Types.arg_label * Ident.t) list;
  body : expression;
  env : env;
}

and env = (Ident.t * binding) list

type state = {
  mutable next : int;
  mutable body : K.instruction list;
  (* Inlining depth, so a recursion whose base case never folds fails with a
     source location instead of running the adapter out of stack. *)
  mutable depth : int;
  (* Total operations emitted. Depth alone does not bound the work: a tree
     recursion with a constant bound stays shallow while inlining
     exponentially many calls, and a nest of constant loops multiplies
     without any single loop exceeding its own limit. This counts what
     actually grows. *)
  mutable emitted : int;
  (* Structural recursion depth of [expression]. Deeply nested expressions
     emit almost nothing, so the operation budget cannot see them, and
     catching Stack_overflow in native code is not dependable. Count instead. *)
  mutable nesting : int;
}

let create_state () = { next = 0; body = []; depth = 0; emitted = 0; nesting = 0 }

(* Limits on compile-time evaluation, overridable like a compiler flag.

   These are not a style policy. A kernel that is expensive on purpose is the
   author's business, the same way a C++ compiler will happily instantiate a
   template that produces a megabyte of code. What a compiler does owe is
   termination and a diagnostic instead of a crash, so each limit below
   answers one of those and nothing else:

   - depth, because a recursion whose base case never folds does not
     terminate, and that is the compiler's problem rather than the author's;
   - nesting, because exhausting the adapter's own stack gives an uncatchable
     runtime error with no source location at all;
   - operations, as a runaway backstop only. The default matches gcc's
     -fconstexpr-ops-limit, so anything an author wrote deliberately fits and
     only genuine exponential blowup trips it.

   For comparison: clang -fconstexpr-depth 512, -fbracket-depth 256,
   -fconstexpr-steps 1048576; gcc -fconstexpr-ops-limit 33554432;
   rustc recursion_limit 128. Tight on depth, generous on operations. *)

(* A malformed limit is refused rather than quietly replaced by the default:
   someone who sets one of these asked for a specific number, and silently
   compiling at a different one is the wrong kind of help. *)
let env_int name default =
  match Sys.getenv_opt name with
  | None -> default
  | Some text ->
      (match int_of_string_opt text with
       | Some n when n > 0 -> n
       | Some _ | None ->
           prerr_endline
             (name ^ " must be a positive integer, got: " ^ text);
           exit 2)

let max_inline_depth = env_int "OXGPU_MAX_INLINE_DEPTH" 64
let max_emitted_operations = env_int "OXGPU_MAX_OPERATIONS" 33554432
let max_expression_nesting = env_int "OXGPU_MAX_NESTING" 8192

(* Constant [for] loops up to this many iterations are copied, not emitted. *)
let unroll_bounds = 64

let fresh s = let n = s.next in s.next <- n + 1; n

let emit_at s exp op =
  if s.emitted >= max_emitted_operations then
    fail exp.exp_loc
      (Printf.sprintf
         "kernel exceeded %d compile-time operations (OXGPU_MAX_OPERATIONS); \
          a recursion with exponential breadth inlines without bound even \
          while its depth stays small"
         max_emitted_operations);
  s.emitted <- s.emitted + 1;
  s.body <- K.instruction ~loc:(span exp.exp_loc) op :: s.body

(* Define one value: append the instruction that computes it and hand back its
   destination as an operand. Nearly everything the primitives do is this
   motion, so the id stays local to the instruction that defines it. *)
let value_at s exp node =
  let d = fresh s in
  emit_at s exp (node d);
  K.Value d

(* Run [f] against an empty instruction list and return what it emitted, in
   order, alongside its result. The enclosing list is restored. *)
let capture_region s f =
  let outer = s.body in
  s.body <- [];
  let result = f () in
  let emitted = List.rev s.body in
  s.body <- outer;
  emitted, result

(* ---- Integer ranges ---- *)

let i32_min = -2147483648L
let i32_max = 2147483647L
let full_range = Some (i32_min, i32_max)

(* A single-interval range collapsed to the constant it pins, or [None] when
   the value is not known at compile time. *)
let constant_range = function
  | Some (lo, hi) when lo = hi -> Some lo
  | _ -> None

let checked loc (lo, hi) =
  if lo < i32_min || hi > i32_max then
    fail loc
      "native int arithmetic may overflow the supported i32 range; use explicit Int32 arithmetic"
  else Some (lo, hi)

(* Two's-complement wrap into signed i32. Products of i32 values fit in int64. *)
let i32_wrap n =
  let two32 = 4294967296L in
  let two31 = 2147483648L in
  let u = Int64.rem n two32 in
  let u = if u < 0L then Int64.add u two32 else u in
  if u >= two31 then Int64.sub u two32 else u

(* ---- Coercions from [binding] ---- *)

let scalar loc = function
  | Scalar (a, r) -> a, r
  | Const_bool (a, _) -> a, None
  | _ -> fail loc "expected scalar"
let memory loc = function Memory m -> m | _ -> fail loc "expected GPU buffer"
let shared loc = function Shared a -> a | _ -> fail loc "expected CTA shared buffer"
let mbarrier loc = function Mbarrier a -> a | _ -> fail loc "expected mbarrier"
let tensor_map loc = function Tensor_map a -> a | _ -> fail loc "expected tensor_map"
let wgmma_acc loc = function Wgmma_acc ids -> ids | _ -> fail loc "expected wgmma_acc"
let unit loc = function Void -> () | _ -> fail loc "expected unit"

let const_extent loc v =
  match scalar loc v with
  | _, Some (lo, hi) when lo = hi && lo > 0L && lo <= Int64.of_int max_int ->
      Int64.to_int lo
  | _ -> fail loc "shared/descriptor extents must be positive compile-time constants"

let const_nat loc v =
  match scalar loc v with
  | _, Some (lo, hi) when lo = hi && lo >= 0L && lo <= Int64.of_int max_int ->
      Int64.to_int lo
  | _ -> fail loc "expected a non-negative compile-time constant"

let lookup env id =
  List.find_map
    (fun (key, v) -> if Ident.same key id then Some v else None)
    env

(* ---- Call arguments ---- *)

type call_args = (Types.arg_label * apply_arg) list

let positional loc (xs : call_args) =
  List.map
    (function
      | Types.Nolabel, Arg (e, _) -> e
      | _ -> fail loc "only positional fully applied calls are supported")
    xs

(* Labelled arguments by name, in the order of [required], plus the trailing
   unlabelled argument if any (the [()] that terminates a labelled call). *)
let labelled loc ~required (xs : call_args) =
  let tbl = Hashtbl.create 8 in
  let unit_arg = ref None in
  List.iter
    (function
      | Types.Labelled name, Arg (e, _) -> Hashtbl.replace tbl name e
      | Types.Nolabel, Arg (e, _) -> unit_arg := Some e
      | Types.Optional _, _ -> fail loc "optional arguments unsupported"
      | _ -> fail loc "unsupported argument form")
    xs;
  let get name =
    match Hashtbl.find_opt tbl name with
    | Some e -> e
    | None -> fail loc ("missing labelled argument ~" ^ name)
  in
  List.map get required, !unit_arg

(* ---- Primitive handlers ---- *)

(* What a primitive sees: the tracing state, the environment, the call being
   lowered, the primitive's name, and [recur] to evaluate a subexpression in
   any environment. Handlers never call the expression evaluator directly, so
   they can live below it in the dependency order. *)
type ctx = {
  s : state;
  env : env;
  exp : expression;
  op : string;
  recur : env -> expression -> binding;
}

type handler = ctx -> call_args -> binding

let emit ctx node = emit_at ctx.s ctx.exp node
let value ctx node = value_at ctx.s ctx.exp node

(* Fresh ids for an instruction that defines several results. *)
let fresh_four ctx =
  let a = fresh ctx.s in
  let b = fresh ctx.s in
  let c = fresh ctx.s in
  let d = fresh ctx.s in
  a, b, c, d

let produce ?range ctx node = Scalar (value ctx node, range)

let bool_const ctx v =
  Const_bool (value ctx (fun d -> K.Const_bool (d, v)), v)

(* Evaluate one argument expression and coerce it to what the operation needs.
   Tracing an argument appends instructions, so callers sequence these with
   [let ... in] rather than [and]. *)
let eval ctx e = ctx.recur ctx.env e
let arg ctx classify e = classify e.exp_loc (eval ctx e)
let num ctx e = fst (arg ctx scalar e)
let unit_arg ctx e = arg ctx unit e

let fail_here ctx text = fail ctx.exp.exp_loc text
let arity ctx = fail_here ctx ("unsupported GPU call/arity: " ^ ctx.op)

(* A primitive whose arguments are all positional. [f] matches the list and
   ends with [| _ -> arity ctx]. *)
let on_positional f : handler =
 fun ctx xs -> f ctx (positional ctx.exp.exp_loc xs)

(* A statement taking only [()] that emits one fixed instruction. *)
let unit_effect node : handler =
  on_positional (fun ctx -> function
    | [u] -> unit_arg ctx u; emit ctx node; Void
    | _ -> arity ctx)

(* Index arithmetic the compiler emits on the author's behalf: tile maps and
   accumulator coordinates, which are hardware layout rather than anything the
   kernel wrote. Bundling the operators means a use site destructures the
   handful it needs instead of restating the [K] constructors. *)
type index_arith = {
  add : K.atom -> K.atom -> K.atom;
  sub : K.atom -> K.atom -> K.atom;
  mul : K.atom -> K.atom -> K.atom;
  div : K.atom -> K.atom -> K.atom;
  rem : K.atom -> K.atom -> K.atom;
  min_ : K.atom -> K.atom -> K.atom;
  konst : int -> K.atom;
}

let index_arith ctx =
  let bin node a b = value ctx (fun d -> node d a b) in
  { add = bin (fun d x y -> K.Add_i32 (d, x, y));
    sub = bin (fun d x y -> K.Sub_i32 (d, x, y));
    mul = bin (fun d x y -> K.Mul_i32 (d, x, y));
    div = bin (fun d x y -> K.Div_i32 (d, x, y));
    rem = bin (fun d x y -> K.Rem_i32 (d, x, y));
    min_ = bin (fun d x y -> K.Min_i32 (d, x, y));
    konst = (fun n -> value ctx (fun d -> K.Const_i32 (d, n))) }

let thread_idx ctx = value ctx (fun d -> K.Thread_idx_x d)
