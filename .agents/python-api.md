# Python API and torch operators

## Follow the existing layers

- `api.py` owns dense encoded weights and functional operations. `moe.py` owns
  expert weights and routing operations. `nn.py` provides module wrappers.
- `_library.py` loads native libraries and registers fake operators.
  `_jit.py` builds missing libraries. Read the [build guide](build-release.md)
  when changing either loader or JIT behavior.
- `reference.py` is a format and numerical reference that works without native
  libraries. Keep it independent of optimized implementations.
- `csrc/torch/ops.cpp` is shared by CUDA and HIP. Keep backend details behind
  the existing native interface instead of creating a second Python API.

Paths above are relative to `python/nf12/` unless they start with `csrc/`.

## Contracts to preserve

- `import nf12` must not load or build native libraries. First native use may
  trigger JIT. Loading and operator registration must remain safe under
  concurrent callers, including free-threaded Python.
- Preserve serialization keys and tensor metadata from the spec. Check that
  reconstructed weights retain the intended device and shared storage.
- Keep leading-dimension behavior, dtype validation, and empty-input handling
  consistent with the public API. Do not silently cast weights to hide an
  unsupported input.
- ROCm PyTorch also uses `torch.cuda` and tensors with `is_cuda=True`. Use the
  existing backend detection rather than treating those names as NVIDIA-only.
- Operator schema changes must update Python wrappers, native registration,
  and fake implementations together. Fake outputs must describe real shapes,
  dtypes, devices, and dynamic dimensions.
- Keep inference compatible with `torch.compile` and GPU graph replay. Do not
  add `.item()`, CPU transfers, or Python locks to traced hot paths.
- Preserve tied embedding/head storage and the `static_weights` readiness
  contract. Avoid full-weight decoding or copies where the operation currently
  keeps weights compressed or decodes bounded slices.
- Follow the existing stable torch ABI boundary. New native APIs must remain
  available at the minimum torch version declared by the build.

Run the relevant shared API cases from `tests/api`. Changes to fake operators
or loading also need an actual compiled or graph-replayed inference call,
whichever path changed. See [validation.md](validation.md).
