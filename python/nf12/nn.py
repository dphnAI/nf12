# SPDX-License-Identifier: Apache-2.0
"""Drop-in inference replacements for bf16 ``nn.Linear`` and ``nn.Embedding``.

Both hold an :class:`~nf12.NF12Weight` (``[N, K]``). Tied weights, such as an
embedding table that is also the ``lm_head``, are one ``NF12Weight`` shared by
both modules. Construct ``NF12Embedding.from_weight(w)`` and
``NF12Linear.from_weight(w)`` with the same ``w``.
"""

from __future__ import annotations

import sys

import torch
from torch import nn

from ._library import load
from .api import ESCAPES_KEY, EXP_BASE_KEY, PACKED_KEY, NF12Weight, check_weight_shape, encode

def verify_weights(model) -> None:
    """Checks every NF12 weight in ``model`` is loaded and consistent (its packed
    array's escaped groups match its escape rows). Call it after assembling
    weights from partial (e.g. sharded) loads. One synchronization; raises
    ValueError naming each problem. MLX models are checked by
    :func:`nf12.mlx.verify_weights`."""
    if not isinstance(model, nn.Module):
        mlx_nn = sys.modules.get("mlx.nn")
        if mlx_nn is not None and isinstance(model, mlx_nn.Module):
            from . import mlx

            return mlx.verify_weights(model)
        raise TypeError(f"nf12: verify_weights needs a torch or MLX module, got {type(model).__name__}")
    problems, counts = [], []
    for name, module in model.named_modules():
        if not isinstance(module, _NF12WeightModule):
            continue
        where = f"{name}.weight" if name else "weight"
        if module.nf12_packed.device.type == "meta":
            problems.append(f"{where}: not loaded (still on the meta device)")
        else:
            counts.append((where, escaped_groups(module.nf12_packed), module.nf12_escapes.shape[0]))
    if counts:
        device = counts[0][1].device
        groups = torch.stack([c.to(device) for _, c, _ in counts]).tolist()  # the one synchronization
        problems += [f"{where}: {PACKED_KEY} has {g} escaped groups but {ESCAPES_KEY} has {rows} rows"
                     for (where, _, rows), g in zip(counts, groups) if g != rows]
    if problems:
        raise ValueError("nf12: NF12 weights failed verification: " + "; ".join(problems))


def escaped_groups(packed: torch.Tensor) -> torch.Tensor:
    """Number of escaped groups in a packed weight (the groups whose 32-bit meta
    word is the M == 0 sentinel: all four bytes zero), as a 0-d tensor on the
    weight's device; a consistent weight has that many escape rows."""
    meta = packed.reshape(-1, 3, 128, 4)[:, 0]  # super-atoms of three 512-byte planes; plane 0 holds the meta words
    return (meta == 0).all(dim=-1).sum()


# One super-atom is 1536 bytes: a 512-byte meta plane, then two low-byte planes.
_SUPER_ATOM_PLANES = (3, 512)
# Both sign bits set and zero offsets encode -0.0 without escapes at exp_base 0.
# +0.0 would hit the M == 0 escape sentinel.
_NEG_ZERO_META = 0xC0


def _check_bf16(w: torch.Tensor) -> None:
    if w.dtype != torch.bfloat16:
        raise ValueError(f"nf12: NF12 encodes bfloat16 weights losslessly; got {w.dtype} (call .bfloat16() first)")


class _NF12WeightModule(nn.Module):
    """Owns an NF12 ``[N, K]`` weight as buffers, saved under SPEC.md section 5
    names (``weight.nf12_packed``, ``weight.nf12_escapes``, ``weight.nf12_exp_base``).
    A fresh module holds an all ``-0.0`` weight."""

    _SHAPE_NAME = "(N, K)"

    def __init__(self, n: int, k: int, device=None) -> None:
        check_weight_shape(n, k)
        super().__init__()
        load()
        packed = torch.zeros((n, k * 3 // 2), dtype=torch.uint8, device=device)
        packed.view(-1, *_SUPER_ATOM_PLANES)[:, 0] = _NEG_ZERO_META
        self.register_buffer("nf12_packed", packed)
        self.register_buffer("nf12_escapes", torch.empty((0, 8), dtype=torch.int16, device=device))
        self.exp_base = 0

    @property
    def weight(self) -> NF12Weight:
        return NF12Weight(self.nf12_packed, self.nf12_escapes, self.exp_base)

    def _assign_weight(self, weight: NF12Weight) -> None:
        self.nf12_packed = weight.packed
        self.nf12_escapes = weight.escapes
        self.exp_base = weight.exp_base

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        destination.update(self.weight.state_dict(prefix + "weight."))

    def _extra_state_keys(self, prefix: str) -> set[str]:
        """Non-weight state-dict keys (e.g. bias) loaded by the subclass."""
        return set()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        weight_prefix = prefix + "weight."
        weight_keys = [weight_prefix + key for key in (PACKED_KEY, ESCAPES_KEY, EXP_BASE_KEY)]
        expected = set(weight_keys) | self._extra_state_keys(prefix)

        missing = [key for key in sorted(expected) if key not in state_dict]
        missing_keys.extend(missing)
        if strict:
            unexpected_keys.extend(key for key in state_dict if key.startswith(prefix) and key not in expected)

        supplied = {key: state_dict[key] for key in weight_keys if key in state_dict}
        if not supplied:
            return
        device = self.nf12_packed.device
        partial = len(supplied) < len(weight_keys)
        if partial and device.type == "meta":
            # No loaded weight to complete a partial one from; the missing keys are
            # reported (and fail a strict load).
            return
        # torch passes strict=True to every module and applies the caller's
        # strictness afterwards from the key lists, so `strict` cannot tell the
        # two apart here. As with torch's own modules, supplied entries are
        # applied and a strict load then reports any missing ones. A weight
        # assembled from partial loads (e.g. split across shards) is consistent
        # only once all of its tensors are in: check it with verify_weights().
        try:
            for key, value in supplied.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"{key} must be a tensor, got {type(value).__name__}")
            merged = supplied
            if partial:  # missing entries keep their current values
                merged = {key: value.to(device) for key, value in
                          {**self.weight.state_dict(weight_prefix), **supplied}.items()}
            weight = NF12Weight.from_state_dict(merged, weight_prefix)
        except ValueError as e:
            error_msgs.append(f"{weight_prefix[:-1]}: {e}")
            return
        if weight.shape != self.weight.shape:
            error_msgs.append(
                f"{weight_prefix[:-1]}: shape {weight.shape} does not match {self._SHAPE_NAME} = {self.weight.shape}"
            )
            return
        if device.type != "meta":
            weight = weight.to(device)
        if weight_keys[0] in supplied and weight_keys[1] in supplied:
            # Supplied together, packed and escapes must agree: rejects corrupt or truncated files.
            groups = int(escaped_groups(weight.packed))
            if groups != weight.num_escapes:
                error_msgs.append(f"{weight_prefix[:-1]}: {PACKED_KEY} has {groups} escaped groups but "
                                  f"{ESCAPES_KEY} has {weight.num_escapes} rows")
                return
        self._assign_weight(weight)


class NF12Linear(_NF12WeightModule):
    """``y = x @ W.T + b`` with an NF12-encoded weight (bf16 activations and output).

    A freshly constructed layer holds an all ``-0.0`` weight. Fill it with
    :meth:`from_linear`, :meth:`from_weight` or ``load_state_dict``. The state
    dict stores the weight under SPEC.md section 5 names
    (``weight.nf12_packed``, ``weight.nf12_escapes``, ``weight.nf12_exp_base``).
    """

    _SHAPE_NAME = "(out_features, in_features)"

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None) -> None:
        super().__init__(out_features, in_features, device)
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.bfloat16, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_weight(cls, weight: NF12Weight, bias: torch.Tensor | None = None) -> NF12Linear:
        """A layer using ``weight`` itself (not a copy), so it can be shared, e.g. with an :class:`NF12Embedding`."""
        n, k = weight.shape
        layer = cls(k, n, bias=bias is not None, device="meta")
        layer._assign_weight(weight)
        if bias is not None:
            if bias.dtype != torch.bfloat16 or bias.shape != (n,):
                raise ValueError(f"nf12: bias must be bfloat16 [{n}], got {bias.dtype} {tuple(bias.shape)}")
            layer.bias = nn.Parameter(bias.detach().to(weight.device), requires_grad=False)
        return layer

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> NF12Linear:
        """Losslessly converts a bf16 CUDA or MPS ``nn.Linear``."""
        w = linear.weight.detach()
        _check_bf16(w)
        return cls.from_weight(encode(w), linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.nf12.linear(x, self.nf12_packed, self.nf12_escapes, self.exp_base, self.bias)

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def _extra_state_keys(self, prefix: str) -> set[str]:
        return {prefix + "bias"} if self.bias is not None else set()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        bias_key = prefix + "bias"
        if self.bias is None or bias_key not in state_dict:
            return
        bias = state_dict[bias_key]
        if bias.shape != self.bias.shape:
            error_msgs.append(f"{bias_key}: shape {tuple(bias.shape)} does not match {tuple(self.bias.shape)}")
        elif self.bias.device.type == "meta":
            self.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)
        else:
            with torch.no_grad():
                self.bias.copy_(bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"exp_base={self.exp_base}, escapes={self.nf12_escapes.shape[0]}"
        )


class NF12Embedding(_NF12WeightModule):
    """``out = W[ids]`` with an NF12-encoded ``[num_embeddings, embedding_dim]`` table (SPEC.md section 10).

    Lookups preserve the BF16 table values. IDs outside ``[0, num_embeddings)``
    give ``+0.0`` rows (vocab-parallel shards). ``ids`` may be int32 or int64 of
    any shape. Construction and state dict work as for :class:`NF12Linear`.
    """

    _SHAPE_NAME = "(num_embeddings, embedding_dim)"

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None) -> None:
        super().__init__(num_embeddings, embedding_dim, device)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    @classmethod
    def from_weight(cls, weight: NF12Weight) -> NF12Embedding:
        """A table using ``weight`` itself (not a copy), so it can be shared, e.g. with a tied :class:`NF12Linear`."""
        n, k = weight.shape
        layer = cls(n, k, device="meta")
        layer._assign_weight(weight)
        return layer

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> NF12Embedding:
        """Losslessly converts a bf16 CUDA or MPS ``nn.Embedding``. Rejects ``max_norm``, which rewrites rows."""
        if embedding.max_norm is not None:
            raise ValueError("nf12: NF12Embedding does not support max_norm")
        w = embedding.weight.detach()
        _check_bf16(w)
        return cls.from_weight(encode(w))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.ops.nf12.embedding(ids, self.nf12_packed, self.nf12_escapes, self.exp_base)

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, exp_base={self.exp_base}, "
            f"escapes={self.nf12_escapes.shape[0]}"
        )
