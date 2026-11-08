# SPDX-License-Identifier: Apache-2.0
"""NF12 on MLX arrays (Apple GPUs).

The same API as the top-level PyTorch functions (``encode``, ``linear``,
``fused_moe`` ...), on ``mlx.core.array``. The kernels are the ones the PyTorch
MPS backend runs (``nf12.metallib``), encoded on MLX's command encoder by the
``nf12._mlx_ext`` extension, so results match ``device="mps"`` bit for bit and
encoded weights move freely between MLX, MPS and CUDA. Ops are lazy MLX graph
nodes; only encoding synchronizes. This module does not import PyTorch.

``nf12.encode(mx_array)`` and the other top-level functions route MLX arrays
here, so either entry point works.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
import threading
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn

PACKED_KEY = "nf12_packed"
ESCAPES_KEY = "nf12_escapes"
EXP_BASE_KEY = "nf12_exp_base"
ESCAPE_OFFSETS_KEY = "nf12_escape_offsets"

_lock = threading.Lock()
_ext = None


def _load():
    """The ``nf12._mlx_ext`` extension, rebuilt in the JIT cache when missing or
    built against another MLX version (MLX has no stable C++ ABI)."""
    global _ext
    if _ext is not None:
        return _ext
    with _lock:
        if _ext is not None:
            return _ext
        ext = None
        try:
            ext = importlib.import_module("nf12._mlx_ext")
        except ImportError:
            pass
        if ext is None or getattr(ext, "mlx_version", None) != mx.__version__:
            from ._jit import mlx_library_directory

            path = mlx_library_directory() / _extension_filename()
            spec = importlib.util.spec_from_file_location("nf12._mlx_ext", path)
            ext = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ext)
            sys.modules["nf12._mlx_ext"] = ext
        try:
            ext._accepts_array(mx.array(0))
        except TypeError as error:
            raise ImportError(
                f"nf12: {ext.__file__} cannot take mx.array arguments: it was built with a different nanobind "
                f"ABI than MLX {mx.__version__}. Install the nanobind version MLX was built with and rebuild "
                "(clear ~/.cache/nf12 for a first-use build)."
            ) from error
        _ext = ext
        return _ext


def _extension_filename() -> str:
    import sysconfig

    return "_mlx_ext" + sysconfig.get_config_var("EXT_SUFFIX")


def check_weight_shape(n: int, k: int) -> None:
    if n <= 0 or n % 16:
        raise ValueError(f"nf12: out_features (N) must be a positive multiple of 16, got {n}")
    if k <= 0 or k % 64:
        raise ValueError(f"nf12: in_features (K) must be a positive multiple of 64, got {k}")


def _check_array(x, name: str) -> None:
    if not isinstance(x, mx.array):
        raise ValueError(f"nf12: {name} must be an mlx.core.array, got {type(x).__name__}")


def _check_bf16_matrix(w: mx.array, name: str) -> None:
    _check_array(w, name)
    if w.dtype != mx.bfloat16 or w.ndim != 2:
        raise ValueError(f"nf12: {name} must be a 2-D bfloat16 array, got {w.dtype} with shape {tuple(w.shape)}")
    check_weight_shape(*w.shape)


@dataclasses.dataclass(eq=False)
class NF12Weight:
    """An NF12-encoded ``[N, K]`` BF16 matrix held in MLX arrays (SPEC.md section 5)."""

    packed: mx.array  # uint8 [N, K*3/2], tiled layout
    escapes: mx.array  # int16 [E, 8], BF16 bit patterns of escaped groups
    exp_base: int  # 0..120
    shape: tuple[int, int] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        p, e = self.packed, self.escapes
        _check_array(p, "packed")
        _check_array(e, "escapes")
        if p.dtype != mx.uint8 or p.ndim != 2 or p.shape[1] % 3:
            raise ValueError(f"nf12: packed must be uint8 [N, K*3/2], got {p.dtype} {tuple(p.shape)}")
        if e.dtype != mx.int16 or e.ndim != 2 or e.shape[1] != 8:
            raise ValueError(f"nf12: escapes must be int16 [E, 8], got {e.dtype} {tuple(e.shape)}")
        self.exp_base = int(self.exp_base)
        if not 0 <= self.exp_base <= 120:
            raise ValueError(f"nf12: exp_base must be in [0, 120], got {self.exp_base}")
        self.shape = (p.shape[0], p.shape[1] * 2 // 3)
        check_weight_shape(*self.shape)

    @property
    def num_escapes(self) -> int:
        return self.escapes.shape[0]

    @property
    def nbytes(self) -> int:
        """Bytes of encoded storage (packed + escapes)."""
        return self.packed.nbytes + self.escapes.nbytes

    def decode(self) -> mx.array:
        return decode(self)

    def state_dict(self, prefix: str = "") -> dict[str, mx.array]:
        """Arrays under SPEC.md section 5 names, e.g. for ``mx.save_safetensors``."""
        return {
            prefix + PACKED_KEY: self.packed,
            prefix + ESCAPES_KEY: self.escapes,
            prefix + EXP_BASE_KEY: mx.array([self.exp_base], dtype=mx.uint8),
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, mx.array], prefix: str = "") -> NF12Weight:
        exp_base = state_dict[prefix + EXP_BASE_KEY]
        if exp_base.dtype != mx.uint8 or exp_base.size != 1:
            raise ValueError(f"nf12: {prefix + EXP_BASE_KEY} must be a uint8 [1] array")
        return cls(state_dict[prefix + PACKED_KEY], state_dict[prefix + ESCAPES_KEY], int(exp_base.item()))


def encode(w: mx.array) -> NF12Weight:
    """Losslessly encodes a ``[N, K]`` bf16 matrix. Evaluates ``w`` and synchronizes."""
    _check_bf16_matrix(w, "weight")
    return NF12Weight(*_load().encode(w))


def decode(weight: NF12Weight) -> mx.array:
    """Recovers the original ``[N, K]`` BF16 matrix."""
    return _load().decode(weight.packed, weight.escapes, weight.exp_base)


def _check_input(x: mx.array, k: int) -> None:
    _check_array(x, "x")
    if x.dtype != mx.bfloat16:
        raise ValueError(f"nf12: x must be bfloat16, got {x.dtype}")
    if x.ndim == 0 or x.shape[-1] != k:
        raise ValueError(f"nf12: x must have shape [..., {k}], got {tuple(x.shape)}")


def linear(x: mx.array, weight: NF12Weight, bias: mx.array | None = None) -> mx.array:
    """``y = x @ W.T (+ bias)``: bf16 ``[..., K]`` -> ``[..., N]``.

    Up to 16 rows run fused decode + multiply kernels (bit-identical to
    :func:`linear_bf16t`); larger inputs decode the weight and use ``mx.matmul``
    (SPEC.md 6.5)."""
    _check_input(x, weight.shape[1])
    return _load().linear(x, weight.packed, weight.escapes, weight.exp_base, bias)


def embedding(ids: mx.array, weight: NF12Weight) -> mx.array:
    """``W[ids]`` for int32 or int64 ``ids`` of any shape -> bf16 ``[*ids.shape, K]``.
    IDs outside ``[0, N)`` give ``+0.0`` rows (SPEC.md section 10)."""
    _check_array(ids, "ids")
    if ids.dtype not in (mx.int32, mx.int64):
        raise ValueError(f"nf12: ids must be int32 or int64, got {ids.dtype}")
    return _load().embedding(ids, weight.packed, weight.escapes, weight.exp_base)


def pack_bf16t(w: mx.array) -> mx.array:
    """Packs ``[N, K]`` bf16 into the uncompressed bf16t twin layout, uint8 ``[N, 2K]`` (SPEC.md 4.4)."""
    _check_bf16_matrix(w, "weight")
    return _load().pack_bf16t(w)


def linear_bf16t(x: mx.array, w_bf16t: mx.array, bias: mx.array | None = None) -> mx.array:
    """Linear on a bf16t-packed weight; equals :func:`linear` on the fused routes."""
    _check_array(w_bf16t, "w_bf16t")
    _check_input(x, w_bf16t.shape[1] // 2)
    return _load().linear_bf16t(x, w_bf16t, bias)


# ---------------------------------------------------------------------------
# Mixture of experts (SPEC.md section 9).


@dataclasses.dataclass(eq=False)
class NF12MoEWeight:
    """``E`` NF12-encoded ``[N, K]`` expert matrices in MLX arrays (SPEC.md section 9.1)."""

    packed: mx.array  # uint8 [E, N, K*3/2]
    escapes: mx.array  # int16 [S, 8], expert-major
    escape_offsets: mx.array  # int64 [E + 1]
    exp_base: mx.array  # uint8 [E]
    shape: tuple[int, int, int] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        p, e, off, b = self.packed, self.escapes, self.escape_offsets, self.exp_base
        for name, t in (("packed", p), ("escapes", e), ("escape_offsets", off), ("exp_base", b)):
            _check_array(t, name)
        if p.dtype != mx.uint8 or p.ndim != 3 or p.shape[0] < 1 or p.shape[2] % 3:
            raise ValueError(f"nf12: packed must be uint8 [E, N, K*3/2] with E >= 1, got {p.dtype} {tuple(p.shape)}")
        if e.dtype != mx.int16 or e.ndim != 2 or e.shape[1] != 8:
            raise ValueError(f"nf12: escapes must be int16 [S, 8], got {e.dtype} {tuple(e.shape)}")
        num_experts = p.shape[0]
        if off.dtype != mx.int64 or tuple(off.shape) != (num_experts + 1,):
            raise ValueError(f"nf12: escape_offsets must be int64 [{num_experts + 1}], got {off.dtype} {tuple(off.shape)}")
        if b.dtype != mx.uint8 or tuple(b.shape) != (num_experts,):
            raise ValueError(f"nf12: exp_base must be uint8 [{num_experts}], got {b.dtype} {tuple(b.shape)}")
        self.shape = (num_experts, p.shape[1], p.shape[2] * 2 // 3)
        check_weight_shape(*self.shape[1:])

    @property
    def num_experts(self) -> int:
        return self.shape[0]

    @property
    def num_escapes(self) -> int:
        return self.escapes.shape[0]

    @property
    def nbytes(self) -> int:
        return self.packed.nbytes + self.escapes.nbytes + self.escape_offsets.nbytes + self.exp_base.nbytes

    def decode(self) -> mx.array:
        """Recovers the original ``[E, N, K]`` BF16 expert weights."""
        return _load().moe_decode(self.packed, self.escapes, self.escape_offsets, self.exp_base)

    def narrow(self, start: int, length: int) -> NF12MoEWeight:
        """Experts ``[start, start + length)`` as a standalone weight with escape
        offsets rebased to 0. Synchronizes to read two offsets."""
        if not (0 <= start and length >= 1 and start + length <= self.num_experts):
            raise ValueError(f"nf12: experts [{start}, {start + length}) out of range for {self.num_experts}")
        offsets = self.escape_offsets[start:start + length + 1]
        first, last = int(offsets[0].item()), int(offsets[-1].item())
        return NF12MoEWeight(self.packed[start:start + length], self.escapes[first:last], offsets - first,
                             self.exp_base[start:start + length])

    def state_dict(self, prefix: str = "") -> dict[str, mx.array]:
        return {
            prefix + PACKED_KEY: self.packed,
            prefix + ESCAPES_KEY: self.escapes,
            prefix + ESCAPE_OFFSETS_KEY: self.escape_offsets,
            prefix + EXP_BASE_KEY: self.exp_base,
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, mx.array], prefix: str = "") -> NF12MoEWeight:
        return cls(state_dict[prefix + PACKED_KEY], state_dict[prefix + ESCAPES_KEY],
                   state_dict[prefix + ESCAPE_OFFSETS_KEY], state_dict[prefix + EXP_BASE_KEY])


def _check_experts(w: mx.array) -> None:
    _check_array(w, "weight")
    if w.dtype != mx.bfloat16 or w.ndim != 3 or w.shape[0] < 1:
        raise ValueError(f"nf12: experts must be a bfloat16 [E, N, K] array, got {w.dtype} {tuple(w.shape)}")
    check_weight_shape(*w.shape[1:])


def encode_moe(w: mx.array) -> NF12MoEWeight:
    """Losslessly encodes stacked ``[E, N, K]`` bf16 experts. Synchronizes."""
    _check_experts(w)
    return NF12MoEWeight(*_load().encode_moe(w))


def pack_bf16t_moe(w: mx.array) -> mx.array:
    """Packs ``[E, N, K]`` bf16 experts into the bf16t twin layout, uint8 ``[E, N, 2K]``."""
    _check_experts(w)
    e, n, k = w.shape
    # N % 16 == 0, so expert e's super-atoms are rows [e*N, (e+1)*N) of the stacked matrix.
    return _load().pack_bf16t(w.reshape(e * n, k)).reshape(e, n, 2 * k)


def _prepare_topk_ids(topk_ids: mx.array, num_experts: int) -> mx.array:
    _check_array(topk_ids, "topk_ids")
    if topk_ids.ndim != 2 or topk_ids.dtype not in (mx.int32, mx.int64):
        raise ValueError(f"nf12: topk_ids must be an int32/int64 [T, topk] array, got {topk_ids.dtype} "
                         f"{tuple(topk_ids.shape)}")
    if topk_ids.dtype == mx.int64:
        # Drop before narrowing: 2**32 must not wrap into a valid expert.
        topk_ids = mx.where((topk_ids >= 0) & (topk_ids < num_experts), topk_ids, -1)
    return topk_ids.astype(mx.int32)


def moe_align(topk_ids: mx.array, num_experts: int) -> tuple[mx.array, mx.array]:
    """Sorts assignment ids ``t * topk + j`` stably by expert; returns ``(sorted int32 [T*topk],
    expert_offsets int64 [E + 1])``. See :func:`nf12.moe_align`."""
    return _load().moe_align(_prepare_topk_ids(topk_ids, num_experts), num_experts)


def _check_moe_input(x: mx.array, k: int) -> None:
    _check_array(x, "x")
    if x.dtype != mx.bfloat16 or x.ndim != 2 or x.shape[1] != k:
        raise ValueError(f"nf12: x must be a bfloat16 [rows, {k}] array, got {x.dtype} {tuple(x.shape)}")


def moe_linear(x: mx.array, weight: NF12MoEWeight | mx.array, expert_offsets: mx.array, *,
               gather: mx.array | None = None, gather_div: int = 1, scatter: mx.array | None = None,
               row_scale: mx.array | None = None, zero_fill: bool = False) -> mx.array:
    """Grouped linear over expert-sorted rows; see :func:`nf12.moe_linear`. ``weight`` is an
    :class:`NF12MoEWeight` or a bf16t twin from :func:`pack_bf16t_moe`, with identical results."""
    if isinstance(weight, NF12MoEWeight):
        _check_moe_input(x, weight.shape[2])
        return _load().moe_linear(x, weight.packed, weight.escapes, weight.escape_offsets, weight.exp_base,
                                  expert_offsets, gather, gather_div, scatter, row_scale, zero_fill)
    return moe_linear_bf16t(x, weight, expert_offsets, gather=gather, gather_div=gather_div, scatter=scatter,
                            row_scale=row_scale, zero_fill=zero_fill)


def moe_linear_bf16t(x: mx.array, w_bf16t: mx.array, expert_offsets: mx.array, *,
                     gather: mx.array | None = None, gather_div: int = 1, scatter: mx.array | None = None,
                     row_scale: mx.array | None = None, zero_fill: bool = False) -> mx.array:
    """:func:`moe_linear` on a bf16t twin ``uint8 [E, N, 2K]``."""
    _check_array(w_bf16t, "w_bf16t")
    if w_bf16t.dtype != mx.uint8 or w_bf16t.ndim != 3:
        raise ValueError(f"nf12: w_bf16t must be uint8 [E, N, 2K], got {w_bf16t.dtype} {tuple(w_bf16t.shape)}")
    _check_moe_input(x, w_bf16t.shape[2] // 2)
    return _load().moe_linear_bf16t(x, w_bf16t, expert_offsets, gather, gather_div, scatter, row_scale, zero_fill)


def moe_combine(y: mx.array, topk: int) -> mx.array:
    """``out[t] = sum_j y[t * topk + j]`` in FP32 in increasing ``j``, returning bf16 ``[T, N]``."""
    _check_array(y, "y")
    return _load().moe_combine(y, topk)


def moe_silu_mul(h: mx.array) -> mx.array:
    """SiLU gating of ``h = [gate | up]`` (bf16 ``[A, 2I]``) -> bf16 ``[A, I]``, one BF16 rounding."""
    _check_array(h, "h")
    return _load().moe_silu_mul(h)


def _out_features(w: NF12MoEWeight | mx.array) -> tuple[int, int, int]:
    if isinstance(w, NF12MoEWeight):
        return w.shape
    _check_array(w, "weight")
    if w.dtype != mx.uint8 or w.ndim != 3:
        raise ValueError(f"nf12: expected an NF12MoEWeight or a bf16t uint8 [E, N, 2K], got {w.dtype} "
                         f"{tuple(w.shape)}")
    return w.shape[0], w.shape[1], w.shape[2] // 2


def _moe_weight_args(w: NF12MoEWeight | mx.array) -> tuple:
    if isinstance(w, NF12MoEWeight):
        return w.packed, w.escapes, w.escape_offsets, w.exp_base
    return w, None, None, None


def fused_moe(x: mx.array, w13: NF12MoEWeight | mx.array, w2: NF12MoEWeight | mx.array,
              topk_weights: mx.array, topk_ids: mx.array, activation: str = "silu", *,
              expert_map: mx.array | None = None) -> mx.array:
    """A gated MoE MLP ``sum_j topk_weights[t, j] * W2_e (silu(gate) * up)``; see :func:`nf12.fused_moe`."""
    if activation != "silu":
        raise ValueError(f"nf12: unsupported activation {activation!r} (only 'silu')")
    e13, n13, k13 = _out_features(w13)
    e2, n2, k2 = _out_features(w2)
    if e13 != e2 or n13 != 2 * k2:
        raise ValueError(f"nf12: w13 {(e13, n13, k13)} and w2 {(e2, n2, k2)} need the same E and N13 == 2 * K2")
    if (x.ndim != 2 or topk_ids.ndim != 2 or tuple(topk_weights.shape) != tuple(topk_ids.shape)
            or topk_ids.shape[0] != x.shape[0]):
        raise ValueError(f"nf12: need x [T, K] and topk_weights/topk_ids [T, topk], got {tuple(x.shape)}, "
                         f"{tuple(topk_weights.shape)}, {tuple(topk_ids.shape)}")
    if expert_map is not None:
        if expert_map.ndim != 1 or expert_map.size == 0:
            raise ValueError(f"nf12: expert_map must be a non-empty 1-D array, got {tuple(expert_map.shape)}")
        valid = (topk_ids >= 0) & (topk_ids < expert_map.shape[0])
        topk_ids = mx.where(valid, expert_map[mx.where(valid, topk_ids, 0)], -1)
    topk_ids = _prepare_topk_ids(topk_ids, e13)
    _check_moe_input(x, k13)
    return _load().fused_moe(x, *_moe_weight_args(w13), *_moe_weight_args(w2),
                             topk_weights.astype(mx.float32), topk_ids, expert_map is not None)


# ---------------------------------------------------------------------------
# mlx.nn modules.

# One super-atom is 1536 bytes: a 512-byte meta plane, then two low-byte planes.
# Both sign bits set and zero offsets encode -0.0 without escapes at exp_base 0.
_NEG_ZERO_META = 0xC0


class _NF12WeightModule(nn.Module):
    """Holds an NF12 ``[N, K]`` weight as frozen arrays under SPEC.md section 5
    names (``weight.nf12_packed``, ``weight.nf12_escapes``, ``weight.nf12_exp_base``)."""

    def __init__(self, n: int, k: int) -> None:
        check_weight_shape(n, k)
        super().__init__()
        planes = mx.zeros((n * k // 1024, 3, 512), dtype=mx.uint8)
        planes[:, 0] = _NEG_ZERO_META
        self._assign_weight(NF12Weight(planes.reshape(n, k * 3 // 2), mx.zeros((0, 8), dtype=mx.int16), 0))

    def _assign_weight(self, weight: NF12Weight) -> None:
        self.weight = weight.state_dict()
        # exp_base is also kept as a Python int (a plain attribute, outside the
        # parameter tree), so calls never evaluate an array: that works under
        # mx.compile, where the arrays are placeholders.
        object.__setattr__(self, "_exp_base", weight.exp_base)
        self.freeze()

    def update(self, parameters: dict, strict: bool = True) -> nn.Module:
        """``nn.Module.update`` (also used by ``load_weights``); re-reads exp_base
        (one sync) when the weight is replaced."""
        super().update(parameters, strict)
        if "weight" in parameters:
            object.__setattr__(self, "_exp_base", NF12Weight.from_state_dict(self.weight).exp_base)
        return self

    def load_weights(self, file_or_weights, strict: bool = True) -> nn.Module:
        """``nn.Module.load_weights`` for NF12 checkpoints (e.g. from ``mx.save_safetensors``).

        The escape count differs per weight, so ``weight.*`` arrays are checked
        as an NF12 weight of this module's shape rather than against the current
        array shapes. Other arrays (``bias``) must match their current shapes.
        ``strict`` requires exactly this module's keys."""
        from mlx.utils import tree_flatten

        weights = dict(mx.load(file_or_weights) if isinstance(file_or_weights, str) else file_or_weights)
        current = dict(tree_flatten(self.parameters()))
        if strict and set(weights) != set(current):
            missing, extra = sorted(set(current) - set(weights)), sorted(set(weights) - set(current))
            raise ValueError(f"nf12: checkpoint keys do not match the module (missing {missing}, unexpected {extra})")
        prefix = "weight."
        nf12 = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        if nf12:
            weight = NF12Weight.from_state_dict(nf12)
            if weight.shape != self.nf12_weight.shape:
                raise ValueError(f"nf12: checkpoint weight shape {weight.shape} != module shape {self.nf12_weight.shape}")
        rest = {key: value for key, value in weights.items() if not key.startswith(prefix) and key in current}
        for key, value in rest.items():
            if value.shape != current[key].shape:
                raise ValueError(f"nf12: {key} has shape {tuple(value.shape)}, expected {tuple(current[key].shape)}")
        update = {**rest}
        if nf12:
            update["weight"] = nf12
        self.update(update)
        return self

    @property
    def nf12_weight(self) -> NF12Weight:
        """The current weight, from the module's arrays and the cached exp_base."""
        return NF12Weight(self.weight[PACKED_KEY], self.weight[ESCAPES_KEY], self._exp_base)

    def _weight_args(self) -> tuple[mx.array, mx.array, mx.array]:
        # The exp_base array (not the cached int) goes to the kernels as data, so
        # an mx.compile trace that captured this module's state stays correct
        # after its weight is replaced by one of the same shape.
        w = self.weight
        return w[PACKED_KEY], w[ESCAPES_KEY], w[EXP_BASE_KEY]

    def _linear(self, x: mx.array, bias: mx.array | None) -> mx.array:
        _check_input(x, self.nf12_weight.shape[1])
        return _load().linear(x, *self._weight_args(), bias)


class NF12Linear(_NF12WeightModule):
    """``y = x @ W.T + b`` with an NF12-encoded weight (bf16 activations and output)."""

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__(output_dims, input_dims)
        if bias:
            self.bias = mx.zeros((output_dims,), dtype=mx.bfloat16)
        self.freeze()

    @classmethod
    def from_weight(cls, weight: NF12Weight, bias: mx.array | None = None) -> NF12Linear:
        n, k = weight.shape
        layer = cls(k, n, bias=bias is not None)
        layer._assign_weight(weight)
        if bias is not None:
            if bias.dtype != mx.bfloat16 or tuple(bias.shape) != (n,):
                raise ValueError(f"nf12: bias must be bfloat16 [{n}], got {bias.dtype} {tuple(bias.shape)}")
            layer.bias = bias
            layer.freeze()
        return layer

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> NF12Linear:
        """Losslessly converts a bf16 ``mlx.nn.Linear``."""
        w = linear.weight
        if w.dtype != mx.bfloat16:
            raise ValueError(f"nf12: NF12 encodes bfloat16 weights losslessly; got {w.dtype}")
        return cls.from_weight(encode(w), linear.get("bias"))

    def __call__(self, x: mx.array) -> mx.array:
        return self._linear(x, self.get("bias"))


class NF12Embedding(_NF12WeightModule):
    """``out = W[ids]`` with an NF12-encoded ``[num_embeddings, dims]`` table."""

    def __init__(self, num_embeddings: int, dims: int) -> None:
        super().__init__(num_embeddings, dims)

    @classmethod
    def from_weight(cls, weight: NF12Weight) -> NF12Embedding:
        n, k = weight.shape
        layer = cls(n, k)
        layer._assign_weight(weight)
        return layer

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> NF12Embedding:
        """Losslessly converts a bf16 ``mlx.nn.Embedding``."""
        w = embedding.weight
        if w.dtype != mx.bfloat16:
            raise ValueError(f"nf12: NF12 encodes bfloat16 weights losslessly; got {w.dtype}")
        return cls.from_weight(encode(w))

    def __call__(self, ids: mx.array) -> mx.array:
        _check_array(ids, "ids")
        if ids.dtype not in (mx.int32, mx.int64):
            raise ValueError(f"nf12: ids must be int32 or int64, got {ids.dtype}")
        return _load().embedding(ids, *self._weight_args())

    def as_linear(self, x: mx.array) -> mx.array:
        """``x @ W.T`` with the table as a tied output projection."""
        return self._linear(x, None)
