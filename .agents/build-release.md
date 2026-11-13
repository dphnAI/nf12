# Builds, JIT, and releases

## Development environment

Use `uv` and the existing project environment. Requirements live in
[README.md](../README.md), [pyproject.toml](../pyproject.toml), and the backend
build files. Avoid copying their version or architecture matrices into guides.

For CUDA development, target the GPU you will test instead of rebuilding every
release architecture. This example targets SM120:

```bash
NF12_CUDA_ARCHS=120 uv sync --locked --group test --group bench
```

Change the target to match the device. `NF12_BUILD_NATIVE=OFF` defers compilation
to first native use. It does not remove the need for a compiler when JIT runs.
Use incremental builds and limit `CMAKE_BUILD_PARALLEL_LEVEL` to the host's
available CPU and memory resources.

For HIP, first provision an environment with matching ROCm PyTorch and the
build requirements. Follow [the ROCm setup](../tests/hip/README.md). Do not let
an isolated build replace ROCm PyTorch with a CUDA or CPU wheel. With the build
requirements already installed in `.venv`, build against that environment:

```bash
uv build --wheel --no-sources --no-build-isolation --python .venv/bin/python
```

The root CMake build detects ROCm PyTorch and selects HIP. Use `uv pip` for
changes to an intentionally provisioned ROCm environment, and `uv run --no-sync`
for its commands so project resolution does not replace that torch build.

## Build and loader invariants

- `CMakeLists.txt` owns native targets and installation. `csrc/hip/CMakeLists.txt`
  owns HIP-specific builds. Check both when changing shared native sources.
- Preserve the torch stable ABI and generic Python wheel compatibility.
  Do not introduce a CPython-version-specific extension unintentionally.
- Keep warning settings and architecture-specific compilation flags intact.
  Do not bypass diagnostics or capability probes to make a build pass.
- Wheels must retain the sources needed for JIT, including nested build files.
  Check source installation patterns when adding a file type or build input.
- JIT must select the visible devices and correct backend, serialize concurrent
  builders, and invalidate its cache when relevant inputs change. Failed builds
  must not leave a completion marker that makes partial outputs reusable.
- Check precompiled loading and missing-library JIT separately. Verify a fresh
  build and cache reuse when changing compilation or cache identity.

## Release changes

Read [.github/workflows/release.yml](../.github/workflows/release.yml) and
[tools/build_release.sh](../tools/build_release.sh) before editing the pipeline.
The script runs inside the release container and removes its temporary build
environments. Do not run it against an unrelated host environment.

- Release wheels use the complete configured CUDA target list, independently of
  the build host's GPU. The current workflow builds Linux x86-64 and ARM64 CUDA
  wheels. Source-level HIP support does not imply published ROCm binaries.
- Preserve bundled runtime dependencies and required license files. Use
  `tools/check_release_wheel.py` to check wheel contents and architecture images,
  then load the installed artifact outside the source tree.
- Keep the package and CMake project versions consistent. Pushed `v*` tags must
  match the package version. The workflow publishes through PyPI Trusted
  Publishing and then creates a GitHub release. Manual dispatch builds wheels
  without publishing them.
- Never push a release tag or publish as a validation step. These actions need
  explicit authorization.
