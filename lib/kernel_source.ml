(* Author-facing staged expressions: ordinary lets and operators, with explicit
   GPU operations only at the points where they are needed. *)
type f32 = F32
type 'a gpu_array = Buffer
type buffer = Buffer_arg of int
type index = Index of Kernel_ast.expr
type value = Value of Kernel_ast.expr
type 'a arg = Buffer_f32 : f32 gpu_array arg | Scalar_f32 : f32 arg

type builder = { name : string; args : Kernel_ast.arg_kind list; mutable body_rev : Kernel_ast.stmt list; mutable active : bool }
let current : builder option ref = ref None
let builder () = match !current with Some b when b.active -> b | _ -> failwith "GPU primitive used outside Kernel_source.compile3"
let expr (Value x) = x
let add a b = Value (Kernel_ast.Add_f32 (expr a, expr b))
let mul a b = Value (Kernel_ast.Mul_f32 (expr a, expr b))
module F32 = struct let ( +. ) = add let ( *. ) = mul end
module Gpu = struct
  let thread_idx_x () = Index Kernel_ast.Thread_idx_x
  let load (Buffer_arg arg) (Index i) = Value (Kernel_ast.Load_f32 (arg, i))
  let store (Buffer_arg arg) (Index i) value =
    let b = builder () in b.body_rev <- Kernel_ast.Store_f32 (arg, i, expr value) :: b.body_rev
end

let stage ~name args f =
  let builder = { name; args; body_rev=[]; active=true } in
  if !current <> None then failwith "nested kernel staging is unsupported";
  current := Some builder;
  Fun.protect ~finally:(fun () -> builder.active <- false; current := None) (fun () ->
    f (); { Kernel_ast.name; args=builder.args; body=List.rev builder.body_rev })

let compile3_bbb ~name f = stage ~name
    [Kernel_ast.Buffer Kernel_ast.F32;Kernel_ast.Buffer Kernel_ast.F32;Kernel_ast.Buffer Kernel_ast.F32]
    (fun () -> f (Buffer_arg 0) (Buffer_arg 1) (Buffer_arg 2))
let compile3_bbf ~name f = stage ~name
    [Kernel_ast.Buffer Kernel_ast.F32;Kernel_ast.Buffer Kernel_ast.F32;Kernel_ast.Scalar Kernel_ast.F32]
    (fun () -> f (Buffer_arg 0) (Buffer_arg 1) (Value (Kernel_ast.Arg 2)))
