"""The OpenMP guard refuses mixing native runtimes (checked in fresh subprocesses)."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("first,second", [("torch", "faiss"), ("torch", "lightgbm"), ("faiss", "lightgbm")])
def test_guard_refuses_conflicting_runtime(first, second):
    code = (f"import {first}\nfrom reelrank.runtime import claim, RuntimeConflict\n"
            f"try:\n    claim('{second}')\nexcept RuntimeConflict as e:\n    print('refused')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "refused" in out.stdout, out.stderr


def test_guard_allows_a_single_runtime():
    code = "from reelrank.runtime import claim\nclaim('lightgbm')\nimport lightgbm\nprint('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "ok" in out.stdout, out.stderr
