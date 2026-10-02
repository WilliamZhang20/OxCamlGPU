let compile_ptx kernel =
  let kernel = Optimizer.optimize kernel in
  let target = Ptx_lowering.lower kernel in
  let physical = Physical_ir.physicalize target in
  Ptx.emit_physical physical
