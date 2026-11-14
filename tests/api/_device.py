# SPDX-License-Identifier: Apache-2.0
"""Device selection for the torch-level suites: CUDA when available, else Apple
GPUs (MPS) on macOS. C API, CUDA graph and cuBLAS-routing tests are CUDA only."""

from __future__ import annotations

import pytest
import torch

if torch.cuda.is_available():
    DEV = "cuda"
elif torch.backends.mps.is_available():
    DEV = "mps"
else:
    DEV = "cuda"  # every test is skipped

requires_gpu = pytest.mark.skipif(DEV == "cuda" and not torch.cuda.is_available(), reason="CUDA or MPS required")
cuda_only = pytest.mark.skipif(DEV != "cuda", reason="CUDA only (C API, CUDA graphs or cuBLAS routing)")


def synchronize() -> None:
    if DEV == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()
