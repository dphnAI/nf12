# Validation and benchmarks

## Choose checks for the changed behavior

Use the [build guide](build-release.md) to prepare an environment with the test
or bench dependency group. Run from the repository root. Select individual
files or pytest node IDs before expanding to a full suite.

```bash
uv run --no-sync pytest tests/api/test_format.py -q
uv run --no-sync pytest tests/api -q
```

The first suite checks the reference format and includes device checks when
available. The second covers shared public contracts and can build native
libraries. Do not report hardware-skipped cases as verified.

- Put backend-neutral cases in `tests/api`, reusing its `dense_ops` and
  `moe_pack` fixtures where appropriate. Put HIP-only cases in `tests/hip`.
- Read [the ROCm test instructions](../tests/hip/README.md) before a HIP run.
  `uv run --no-sync pytest tests/api tests/hip -q` must use matching ROCm
  PyTorch and the documented toolchain. Do not bypass its version checks.
- `tests/kernels/` contains architecture-specific CUDA harnesses. Inspect the
  relevant harness before invoking it. Some files are tuning tools rather than
  routine tests.
- ABI changes need the C consumer and descriptor/status tests, not just Python
  calls. Loader or packaging changes need the JIT and installed-wheel checks.

## Correctness evidence

- For a bug, keep a focused failing-before/passing-after regression when
  practical. Test observable outputs or state transitions rather than source
  text, forwarding, or incidental error wording.
- Compare codecs with the independent reference and original BF16 storage.
  Include exception groups and the special values affected by the change.
  Preserve exact assertions where the spec requires equality.
- Use the matched arithmetic reference described in [kernels.md](kernels.md).
  Do not loosen tolerances or remove coverage to make a changed kernel pass.
- Exercise the changed operation through the public API on real hardware.
  Cover non-default streams, graph replay, or concurrent callers when those
  contracts are affected. Unit tests alone do not establish model correctness.
- For shared CUDA/HIP code, run both backends when available. Otherwise record
  the untested backend and the hardware or toolchain needed to validate it.
- Record the source revision and library path used. A run against an old
  installed wheel does not validate newly edited native code.

## Performance and memory

Use existing `bench/` scripts and timing helpers before creating a new harness.
Inspect each script's `--help` for its current options, for example:

```bash
uv run --no-sync python bench/bench_linear.py --help
uv run --no-sync python bench/bench_moe.py --help
```

- Compare before and after on the same GPU with fixed shapes and inputs.
  Record GPU architecture, driver/toolkit, torch version, and exact commands.
- Separate compilation and warmup from steady-state timing. State whether
  measurements use events or graph replay and whether weights are hot or cold
  in cache. Kernel latency is not end-to-end serving throughput.
- Identify the baseline. A BF16-tiled twin checks matched arithmetic, while
  cuBLAS or a serving backend measures a different implementation.
- Use representative batch sizes and escape rates. For MoE, record the expert
  count, top-k, and routing distribution. The benchmark scripts may substitute
  random weights when real weights are missing. Check the recorded weight
  source before describing a result as a model benchmark.
- Include escape storage and temporary decoded weights in memory accounting.
  Distinguish encoded weight bytes, peak allocation, and total device usage.
  The packed payload's 12 bits per weight is not the full storage cost.
- Check free GPU memory and storage before running. Keep results outside tracked
  source, avoid overlapping workloads, and stop only processes you started.
  Request permission before using resources occupied by someone else.

Report commands, measured results, and remaining gaps. Label synthetic inputs
and microbenchmarks explicitly. Do not claim a speedup from code inspection.
