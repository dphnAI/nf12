# SPDX-License-Identifier: Apache-2.0
"""Lazy loading of the native libraries and the fake (meta) op implementations.

Nothing is loaded at ``import nf12`` time, so pure-Python modules such as
``nf12.reference`` work without a compiled build.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import torch

import nf12

_lock = threading.Lock()
_loaded = False

# macOS builds the Apple GPU backend: libnf12_mps.dylib plus nf12.metallib.
METAL = sys.platform == "darwin"
TORCH_LIBRARY = "libnf12_mps.dylib" if METAL else "libnf12_torch.so"
# Files that must sit next to each torch library.
_COMPANIONS = {"libnf12_torch.so": ("libnf12.so",),
               "libnf12_mps.dylib": ("nf12.metallib", "libnf12_mps.torch_version")}
_NATIVE = {"libnf12.so", "libnf12_torch.so", "libnf12_mps.dylib", "nf12.metallib"}


def built_for_this_torch(library: Path) -> bool:
    """Whether a packaged ``libnf12_mps.dylib`` was built against the running torch.

    It uses PyTorch's MPS internals, which have no stable ABI, so CMake records
    the build's torch version next to it. ``libnf12_torch.so`` (CUDA) uses the
    torch stable ABI and loads under any torch >= 2.10, so it is not checked.
    """
    if library.name != "libnf12_mps.dylib":
        return True
    try:
        built = library.with_name("libnf12_mps.torch_version").read_text().strip()
    except OSError:
        return False
    return built == torch.__version__


def library_path(name: str = "libnf12.so") -> Path:
    """Find a packaged native library, or build and cache it for visible GPUs."""
    if METAL and name in ("libnf12.so", "libnf12_torch.so"):
        raise ImportError(f"nf12: {name} (the C API and CUDA torch ops) is CUDA only; macOS builds the Metal backend")
    # Editable installs put the native install tree before the Python source tree.
    for directory in nf12.__path__:
        candidate = Path(directory) / name
        if candidate.is_file():
            if not all(candidate.with_name(c).is_file() for c in _COMPANIONS.get(name, ())):
                continue
            if not built_for_this_torch(candidate):
                continue  # built for another torch: use (or build) the JIT copy instead
            return candidate
    if name in _NATIVE:
        from ._jit import library_directory

        return library_directory() / name
    raise ImportError(f"nf12: native library {name} not found in {list(nf12.__path__)}")


def load() -> None:
    """Idempotently loads the torch extension (``torch.ops.nf12``) on first use."""
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        torch.ops.load_library(str(library_path(TORCH_LIBRARY)))
        _register_fakes()
        _loaded = True


def _register_fakes() -> None:
    @torch.library.register_fake("nf12::encode")
    def _encode(w):
        ctx = torch.library.get_ctx()
        n, k = w.shape
        packed = w.new_empty((n, k * 3 // 2), dtype=torch.uint8)
        escapes = w.new_empty((ctx.new_dynamic_size(), 8), dtype=torch.int16)
        return packed, escapes, ctx.new_dynamic_size(min=0, max=120)

    @torch.library.register_fake("nf12::decode")
    def _decode(packed, escapes, exp_base):
        n, row_bytes = packed.shape
        return packed.new_empty((n, row_bytes * 2 // 3), dtype=torch.bfloat16)

    @torch.library.register_fake("nf12::linear")
    def _linear(x, packed, escapes, exp_base, bias=None):
        return x.new_empty((*x.shape[:-1], packed.shape[0]))

    @torch.library.register_fake("nf12::pack_bf16t")
    def _pack_bf16t(w):
        n, k = w.shape
        return w.new_empty((n, 2 * k), dtype=torch.uint8)

    @torch.library.register_fake("nf12::linear_bf16t")
    def _linear_bf16t(x, w_bf16t, bias=None):
        return x.new_empty((*x.shape[:-1], w_bf16t.shape[0]))

    @torch.library.register_fake("nf12::embedding")
    def _embedding(ids, packed, escapes, exp_base):
        return packed.new_empty((*ids.shape, packed.shape[1] * 2 // 3), dtype=torch.bfloat16)

    @torch.library.register_fake("nf12::encode_moe")
    def _encode_moe(w):
        ctx = torch.library.get_ctx()
        e, n, k = w.shape
        packed = w.new_empty((e, n, k * 3 // 2), dtype=torch.uint8)
        escapes = w.new_empty((ctx.new_dynamic_size(), 8), dtype=torch.int16)
        return packed, escapes, w.new_empty((e + 1,), dtype=torch.int64), w.new_empty((e,), dtype=torch.uint8)

    @torch.library.register_fake("nf12::moe_align")
    def _moe_align(topk_ids, num_experts):
        return (topk_ids.new_empty((topk_ids.numel(),), dtype=torch.int32),
                topk_ids.new_empty((num_experts + 1,), dtype=torch.int64))

    @torch.library.register_fake("nf12::moe_decode")
    def _moe_decode(packed, escapes, escape_offsets, exp_base):
        e, n, row_bytes = packed.shape
        return packed.new_empty((e, n, row_bytes * 2 // 3), dtype=torch.bfloat16)

    def _moe_rows(x, gather, scatter):
        return gather.shape[0] if gather is not None else scatter.shape[0] if scatter is not None else x.shape[0]

    @torch.library.register_fake("nf12::moe_linear")
    def _moe_linear(x, packed, escapes, escape_offsets, exp_base, expert_offsets, gather=None, gather_div=1,
                    scatter=None, row_scale=None, zero_fill=False):
        return x.new_empty((_moe_rows(x, gather, scatter), packed.shape[1]))

    @torch.library.register_fake("nf12::moe_linear_bf16t")
    def _moe_linear_bf16t(x, w_bf16t, expert_offsets, gather=None, gather_div=1, scatter=None, row_scale=None,
                          zero_fill=False):
        return x.new_empty((_moe_rows(x, gather, scatter), w_bf16t.shape[1]))

    @torch.library.register_fake("nf12::moe_combine")
    def _moe_combine(y, topk):
        return y.new_empty((y.shape[0] // topk, y.shape[1]))

    @torch.library.register_fake("nf12::moe_silu_mul")
    def _moe_silu_mul(h):
        return h.new_empty((h.shape[0], h.shape[1] // 2))

    @torch.library.register_fake("nf12::_fused_moe")
    def _fused_moe(x, packed13, escapes13, escape_offsets13, exp_base13,
                   packed2, escapes2, escape_offsets2, exp_base2, topk_weights, topk_ids, zero_fill=False):
        return x.new_empty((x.shape[0], packed2.shape[1]))
