(* Host-side mode check for kernel buffer arguments. Call this before any raw
   CUDA launch; it does not perform the launch or talk to the device ABI. *)
open Gpu_mode
open Ir

type buffer_actual = {
  buffer_id : int;
  ownership : ownership;
  permission : permission;
  locality : locality;
  domain_portability : domain_portability;
  gpu_boundary : gpu_boundary;
}

let to_actual (a : buffer_actual) : actual = {
  buffer_id = a.buffer_id;
  actual_ownership = a.ownership;
  actual_permission = a.permission;
  actual_locality = a.locality;
  actual_domain_portability = a.domain_portability;
  actual_gpu_boundary = a.gpu_boundary;
}

let buffer ?(ownership=Aliased) ?(permission=Read_only) ?(locality=Global)
    ?(domain_portability=Domain_portability_unspecified)
    ?(gpu_boundary=Boundary_portable) buffer_id =
  { buffer_id; ownership; permission; locality; domain_portability; gpu_boundary }

let check kernel actuals =
  Call_verifier.check_call kernel (List.map to_actual actuals)

let check_exn kernel actuals =
  match check kernel actuals with
  | Ok () -> ()
  | Error errors -> raise (Verifier.Invalid_kernel errors)
