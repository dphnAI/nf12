# SPDX-License-Identifier: Apache-2.0
"""Package surface: what `import nf12` provides (no GPU needed)."""

from __future__ import annotations

import subprocess
import sys


def test_submodules_are_attributes_after_import():
    """`import nf12` loads lazily, but nf12.nn / nf12.moe / nf12.api must still
    resolve as attributes, as they did when the package imported them eagerly."""
    code = ("import nf12; assert nf12.nn.NF12Linear and nf12.moe.fused_moe and nf12.api.encode; "
            "assert nf12.reference.encode; assert 'nn' in dir(nf12)")
    subprocess.run([sys.executable, "-c", code], check=True)
