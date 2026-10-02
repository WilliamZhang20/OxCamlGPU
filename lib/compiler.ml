let compile_ptx kernel =
  let kernel = Optimizer.optimize kernel in
  let target = Ptx_lowering.lower kernel in
  Ptx.emit_target target
