open Ir

let reg v = "%r" ^ string_of_int v.id
let freg v = "%f" ^ string_of_int v.id
let ptrreg v = "%rd" ^ string_of_int v.id
let emit k =
  Verifier.verify_exn k;
  let b = Buffer.create 512 in
  let line s = Buffer.add_string b (s ^ "\n") in
  line ".version 8.7";
  line ".target sm_90";
  line ".address_size 64";
  line (".visible .entry " ^ k.name ^ "(");
  List.iteri (fun i a -> line (Printf.sprintf "  .param %s %%arg%d%s" (if a.value.ty = Gpu_type.F32 then ".f32" else ".u64") i (if i + 1 = List.length k.args then "" else ","))) k.args;
  line ")";
  line "{";
  line "  .reg .pred %pred<8>;";
  line "  .reg .b32 %r<256>;";
  line "  .reg .f32 %f<256>;";
  line "  .reg .b64 %rd<256>;";
  List.iteri (fun i a -> line (Printf.sprintf "  ld.param.%s %s, [%%arg%d];" (if a.value.ty = Gpu_type.F32 then "f32" else "u64") (if a.value.ty = Gpu_type.F32 then freg a.value else ptrreg a.value) i)) k.args;
  List.iter (fun i -> match i with
    | Const_i32 (v, n) -> line (Printf.sprintf "  mov.u32 %s, %d;" (reg v) n)
    | Const_f32 (v, f) -> line (Printf.sprintf "  mov.f32 %s, %s;" (freg v) (Printf.sprintf "%.9g" f))
    | Thread_idx_x v -> line (Printf.sprintf "  mov.u32 %s, %%tid.x;" (reg v))
    | Gep_f32 (d,p,ix) ->
        line (Printf.sprintf "  mul.wide.u32 %%rd254, %s, 4;" (reg ix));
        line (Printf.sprintf "  add.u64 %s, %s, %%rd254;" (ptrreg d) (ptrreg p))
    | Add_i32 (d,a,c) -> line (Printf.sprintf "  add.u32 %s, %s, %s;" (reg d) (reg a) (reg c))
    | Add_f32 (d,a,c) -> line (Printf.sprintf "  add.f32 %s, %s, %s;" (freg d) (freg a) (freg c))
    | Mul_f32 (d,a,c) -> line (Printf.sprintf "  mul.f32 %s, %s, %s;" (freg d) (freg a) (freg c))
    | Load_f32 (d,p) -> line (Printf.sprintf "  ld.global.f32 %s, [%s];" (freg d) (ptrreg p))
    | Store_f32 (p,v) -> line (Printf.sprintf "  st.global.f32 [%s], %s;" (ptrreg p) (freg v))
    | Return _ -> line "  ret;") k.body;
  line "  ret;"; line "}"; Buffer.contents b
