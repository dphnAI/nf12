# Kernels and C ABI

## Sources of truth

Read the relevant sections of [SPEC.md](../docs/SPEC.md) before changing format
or arithmetic. Use [ABI_BACKENDS.md](../docs/ABI_BACKENDS.md) for backend
ownership. The main implementation boundaries are:

- `include/nf12/format.cuh` defines the canonical CUDA format helpers.
  `python/nf12/reference.py` supplies the independent, pure-PyTorch reference.
- `include/nf12/nf12.h` defines the public C ABI. `csrc/api.cpp` owns shared
  validation, and `csrc/status.cpp` owns shared status handling.
- `csrc/backend_cuda.cu` and `csrc/hip/backend.hip` implement the private backend
  boundary. Native launch planning and workspace sizes belong to the backend.
- CUDA kernels live under `csrc/codec/`, `csrc/linear/`, and `csrc/moe/`.
  HIP kernels live under `csrc/hip/`. Read its README for RDNA4-specific work.

## Format and numerics

- Preserve exponent-base selection, sentinel handling, and canonical escape
  ordering. Rare values must survive compression. Do not round them or discard
  exception groups to improve compression or speed.
- Treat serialized names and layouts as a compatibility contract. A format
  change requires updating every encoder and consumer, including the reference
  and both backends. Account for existing checkpoints before changing it.
- Check codec recovery through integer views of BF16 storage. Floating-point
  equality cannot distinguish signed zero or verify NaN payloads.
- For fused arithmetic, compare against the BF16-tiled twin with the same
  kernel configuration. For decode-then-GEMM paths, use the same GEMM calls on
  decoded weights. Apply only the tolerances allowed by the spec when comparing
  with a different accumulation order, such as a cuBLAS baseline.
- Preserve FP32 accumulation, specified rounding, and deterministic reduction
  order. Do not enable fast math or unordered floating-point reductions to
  improve a benchmark.

## Execution and ownership

- Keep shared argument validation in the common frontend. Do not duplicate it
  per vendor or add per-call backend discovery to the shared C API.
- Preserve public descriptor layouts, status categories, and error precedence.
  Unsupported hardware or shapes must fail through the existing contract.
- C callers own workspace. Torch operators use the caching allocator. Do not
  introduce shared mutable scratch buffers or hidden allocations in C launches.
- Respect the supplied device and stream. Inference operations must retain
  graph capture and replay support without new host synchronization. Encoding
  has a separate synchronization contract in the spec.
- Preserve buffer lifetimes across asynchronous work. With `static_weights`,
  only ready, immutable weight buffers may be read before the dependency wait.
- Test boundary tiles and dispatch crossovers, including empty inputs and
  non-contiguous inputs where the API permits them. Validate workspace sizing
  against the launch path that consumes it.
- Keep architecture-specific instructions behind capability checks. A generic
  PTX fallback does not establish support for an architecture-specific kernel.
- Fix compiler diagnostics without weakening the existing warning policy.

Use the [validation guide](validation.md) for reference comparisons and runtime
checks. A CUDA result does not validate HIP, or the reverse.
