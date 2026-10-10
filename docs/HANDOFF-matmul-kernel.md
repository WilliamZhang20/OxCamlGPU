# Status: matmul as an OxCaml kernel

Repo: OxCamlGPU
Last verified: 2026-10-09 on an H100 80GB HBM3, CUDA 12.9.

The performant Hopper TMA + WGMMA schedule is OxCaml source at
[`examples/kernels/matmul_tiled.ml`](../examples/kernels/matmul_tiled.ml). It
lowers through the Typedtree adapter to `.gpu` metadata, verified IR, then PTX
(`wgmma.mma_async`, `cp.async.bulk.tensor`, mbarrier, shared wgmma layout).
Nothing matmul-specific lives under `lib/`; authors schedule hierarchy ops
explicitly.

The tile catalog is a module of `test/emit_matmul_ptx.ml`.

For the design, the launch contract, and the two scheduling idioms the kernel
depends on, read [matmul-frontend.md](matmul-frontend.md). For the numbers,
read [measured GEMM performance](architecture.md#measured-gemm-performance).

## Where it stands

- All eight catalog specializations assemble for `sm_90a`, no register spills.
- `test/hardware/run_matmul_h100.sh` checks 256³ against a CPU reference and passes.
- `dune build @runtest` is green.
- 4096³ TF32 holds a repeatable mid-90s percentage of cuBLAS; 8192³ runs at
  parity. Under Nsight Compute both durations match cuBLAS within 1%.

## How to emit the kernel PTX

```bash
source tools/oxcaml-env.sh
tmpdir=$(mktemp -d)
tools/compile_oxcaml_kernels.sh "$tmpdir"
dune exec test/emit_matmul_ptx.exe -- --gpu "$tmpdir/matmul_tiled.gpu"

# Launch facts the PTX text alone does not give you:
#   threads  dynamic_shared_bytes  total_shared_bytes
dune exec test/emit_matmul_ptx.exe -- --info "$tmpdir/matmul_tiled.gpu"
# 288 148112 148112

dune exec test/emit_matmul_ptx.exe -- --print-choose 8192 8192 8192
# matmul_tiled 128 256 32 288
```

The kernel needs 148112 bytes of **dynamic** shared memory. A launcher that
passes 0 will fail, and the kernel cannot be built as a static allocation at
all, since sm_90 caps static `.shared` at 48 KiB.

## Tile parameterization

Tile sizes are parameters of `hopper_gemm`. Each catalog shape is a binding
that applies the schedule to constants, with `[@@gpu.threads]` on that
binding. `choose_config` always picks stages=3; the stages=2 bindings stay
reachable through `--print-config` for comparison.
`matmul_tiled` is 128×256×32, stages=3, group_m=16, 288 threads.
`Matmul_config.kernel_binding` maps a config to the entry name.
`--print-choose` / `--print-config` print `name bm bn bk threads` and emit no
PTX. The barrier ladder stays three deep for both stage counts, because the
stage index is dynamic.

## Files to read first

| Path | Why |
| --- | --- |
| `examples/kernels/matmul_tiled.ml` | The kernel |
| `docs/matmul-frontend.md` | Design, launch contract, scheduling idioms |
| `lib/transform/store_vector.ml` | Why the epilogue vectorizes without being written that way |
| `lib/backend/ptx_ir.ml` | `shared_plan`: the one shared-window layout |
| `tools/export_typedtree_modes.ml` | for / wgmma_acc / shared_wgmma |
| `lib/dsl/gpu_dsl.mli` | Author API |
| `test/emit_matmul_ptx.ml` | `--gpu`, `--info`, `--print-choose` |
