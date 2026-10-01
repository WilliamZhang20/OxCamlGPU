let compile_ptx kernel =
  let kernel = Optimizer.optimize kernel in
  Ptx.emit kernel
