open Gpu_type
open Gpu_ir

let v ?(own=Aliased) ?(loc=Local) ?(perm=Read_only) id ty =
  make_value ~ownership:own ~locality:loc ~permission:perm id ty
let ptr ?(own=Aliased) ?(loc=Global) ?(perm=Read_only) id =
  v ~own ~loc ~perm id (Ptr (F32, Global))
let arg name value = { name; value }

let vector_add =
  let x = ptr 0 and y = ptr 1 and z = ptr ~own:Unique ~perm:Read_write 2 in
  let ix = v 3 I32 and xp = ptr ~loc:Local 4 and yp = ptr ~loc:Local 5
  and zp = ptr ~own:Unique ~loc:Local ~perm:Read_write 6 in
  let xv = v 7 F32 and yv = v 8 F32 and sum = v 9 F32 in
  { name="vector_add"; args=[arg "x" x; arg "y" y; arg "z" z]; body=[
      Thread_idx_x ix; Gep_f32 (xp,x,ix); Gep_f32 (yp,y,ix); Gep_f32 (zp,z,ix);
      Load_f32 (xv,xp); Load_f32 (yv,yp); Add_f32 (sum,xv,yv); Store_f32 (zp,sum)] }

let saxpy =
  let x = ptr 0 and y = ptr ~own:Unique ~perm:Read_write 1 and alpha = v ~loc:Global 2 F32 in
  let ix = v 3 I32 and xp = ptr ~loc:Local 4
  and yp = ptr ~own:Unique ~loc:Local ~perm:Read_write 5 in
  let xv = v 6 F32 and yv = v 7 F32 and ax = v 8 F32 and sum = v 9 F32 in
  { name="saxpy"; args=[arg "x" x; arg "y" y; arg "alpha" alpha]; body=[
      Thread_idx_x ix; Gep_f32 (xp,x,ix); Gep_f32 (yp,y,ix);
      Load_f32 (xv,xp); Load_f32 (yv,yp); Mul_f32 (ax,alpha,xv);
      Add_f32 (sum,ax,yv); Store_f32 (yp,sum)] }
