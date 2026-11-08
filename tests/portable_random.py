# SPDX-License-Identifier: Apache-2.0
"""Seeded test data that is bit-identical on every machine and device.

Device generators are not portable: CUDA's Philox output depends on the GPU's
SM count, and MPS has its own generator. CPU ``torch.randn`` is not portable
either: its vectorized log/cos differ between x86-64, Linux aarch64 and macOS
arm64. CPU ``torch.randint`` and ``torch.rand`` are (integer Mersenne Twister
draws), so everything here is built from them with exact integer arithmetic
and correctly rounded IEEE operations, then moved to the target device.
"""

from __future__ import annotations

import torch

# Four uniform 16-bit integers: 2 * sum - 4 * 65535 + 1 is odd (never zero) with
# standard deviation sqrt(4 * (2^32 - 1) / 3) and |value| <= 262141 < 2^24, so
# it converts to float32 exactly.
_IRWIN_HALL_TERMS = 4
_IRWIN_HALL_STD = (_IRWIN_HALL_TERMS * ((1 << 32) - 1) / 3) ** 0.5


def generator(seed: int | torch.Generator) -> torch.Generator:
    """A CPU generator for ``seed`` (or ``seed`` itself if it is one)."""
    return seed if isinstance(seed, torch.Generator) else torch.Generator().manual_seed(seed)


def randn(*shape: int, seed: int | torch.Generator, scale: float = 1.0, dtype: torch.dtype = torch.bfloat16,
          device: str | torch.device = "cpu") -> torch.Tensor:
    """Approximately normal values (Irwin-Hall with four terms: |x| < 3.47 standard
    deviations, never exactly zero) times ``scale``, in ``dtype`` on ``device``."""
    u = torch.randint(0, 1 << 16, (_IRWIN_HALL_TERMS, *shape), generator=generator(seed), dtype=torch.int32)
    odd = 2 * u.sum(0) - _IRWIN_HALL_TERMS * 65535 + 1
    # One correctly rounded float32 multiply, then one rounding to dtype.
    return (odd.to(torch.float32) * (scale / _IRWIN_HALL_STD)).to(dtype).to(device)


def rand(*shape: int, seed: int | torch.Generator, device: str | torch.device = "cpu") -> torch.Tensor:
    """Uniform float32 values in [0, 1)."""
    return torch.rand(*shape, generator=generator(seed)).to(device)


def randint(low: int, high: int, shape: tuple[int, ...], seed: int | torch.Generator,
            dtype: torch.dtype = torch.int64, device: str | torch.device = "cpu") -> torch.Tensor:
    """Uniform integers in [low, high)."""
    return torch.randint(low, high, shape, generator=generator(seed), dtype=dtype).to(device)


def randperm(n: int, seed: int | torch.Generator, device: str | torch.device = "cpu") -> torch.Tensor:
    """A random permutation of 0 .. n - 1 (int64)."""
    return torch.randperm(n, generator=generator(seed)).to(device)


def normalized(r: torch.Tensor) -> torch.Tensor:
    """Rows of ``r`` divided by their sum, accumulated column by column (no
    reduction-order or exp dependence)."""
    total = r[:, 0].clone()
    for j in range(1, r.shape[1]):
        total = total + r[:, j]
    return r / total.unsqueeze(1)
