#!/usr/bin/env python3
"""Run the full test matrix for this machine and print one summary.

macOS (PyTorch MPS and MLX): the whole suite, then the suites under every
forced kernel route (NF12_MPS_LINEAR, NF12_MPS_MOE), which apply to both
frameworks. Linux (CUDA): the API suite and the release smoke load.

A configuration fails if any test fails or if it ran no tests at all, so a
matrix that silently skips everything cannot pass. Run it on every platform
before pushing changes (see docs/METAL.md, "Testing").

    .venv/bin/python tools/run_tests.py            # full matrix
    .venv/bin/python tools/run_tests.py --quick    # default routes only
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTEST = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]

LINEAR_SUITES = ["tests/api/test_torch.py", "tests/api/test_embedding.py", "tests/api/test_modules.py", "tests/mlx"]
MOE_SUITES = ["tests/api/test_moe.py", "tests/mlx"]


def configurations(quick: bool) -> list[tuple[str, list[str], dict[str, str]]]:
    if platform.system() == "Darwin":
        runs = [("all suites, default routes", PYTEST, {}),
                # Metal API validation: contract violations (e.g. misaligned buffer
                # offsets) that give right answers on one GPU and wrong ones on another.
                ("Metal API validation", PYTEST, {"MTL_DEBUG_LAYER": "1"})]
        if not quick:
            runs += [(f"NF12_MPS_LINEAR={r}", PYTEST + LINEAR_SUITES, {"NF12_MPS_LINEAR": r})
                     for r in ("small", "mma", "gemm", "decode")]
            runs += [(f"NF12_MPS_MOE={r}", PYTEST + MOE_SUITES, {"NF12_MPS_MOE": r}) for r in ("small", "mma", "gemm")]
        return runs
    return [("tests/api", PYTEST + ["tests/api"], {}),
            ("release smoke load", [sys.executable, "tools/check_release_wheel.py", "--load"], {})]


def summarize(output: str) -> tuple[int, int, int, str]:
    """(passed, failed + errors, skipped, summary line) from pytest -q stdout."""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    count = lambda word: sum(int(n) for n in re.findall(rf"(\d+) {word}", last))  # noqa: E731
    return count("passed"), count("failed") + count("errors?"), count("skipped"), last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the forced-route runs")
    args = parser.parse_args()

    results = []
    for name, command, env in configurations(args.quick):
        start = time.perf_counter()
        # Each configuration sets its own routes: drop any the caller exported.
        base = {k: v for k, v in os.environ.items() if not k.startswith(("NF12_MPS_", "MTL_DEBUG"))}
        proc = subprocess.run(command, cwd=ROOT, env={**base, **env}, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        elapsed = time.perf_counter() - start
        if command[1:3] == ["-m", "pytest"]:
            # pytest's summary is the last line of stdout; stderr can end with
            # unrelated warnings or framework logs.
            passed, failed, skipped, last = summarize(proc.stdout)
            ok = proc.returncode == 0 and failed == 0 and passed > 0
            detail = f"{passed} passed, {failed} failed, {skipped} skipped"
        else:
            ok, last = proc.returncode == 0, (output.strip().splitlines() or [""])[-1]
            detail = last
        results.append((name, ok, detail, elapsed))
        print(f"{'PASS' if ok else 'FAIL'}  {name:32s} {detail}  ({elapsed:.0f}s)", flush=True)
        if not ok:
            failures = [line for line in output.splitlines() if line.startswith(("FAILED", "ERROR"))]
            print("\n".join("      " + line for line in failures[:20]) or output[-2000:], flush=True)

    bad = [name for name, ok, _, _ in results if not ok]
    print(f"\n{platform.node()} ({platform.system()} {platform.machine()}): "
          f"{len(results) - len(bad)}/{len(results)} configurations passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
