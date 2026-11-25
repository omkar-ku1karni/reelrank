"""Process-isolation guard for native OpenMP runtimes.

On macOS the PyTorch, FAISS and LightGBM wheels each load their own copy of `libomp`. Measured on an
Apple M4 (see README, "Process isolation"): torch+FAISS segfaults, torch+LightGBM segfaults or deadlocks
and FAISS+LightGBM deadlocks once both have run parallel work in one process. implicit (ALS) is safe with
all three. The pipeline therefore keeps each family in its own process and exchanges `.npy` / Parquet files.

Call `claim("torch")` (etc.) before importing a heavy native library: it raises a clear error instead of
letting the process crash later.
"""

from __future__ import annotations

import os
import sys

FAMILIES = {"torch": "torch", "faiss": "faiss", "lightgbm": "lightgbm"}
CONFLICTS = {
    "torch": {"faiss", "lightgbm"},
    "faiss": {"torch", "lightgbm"},
    "lightgbm": {"torch", "faiss"},
}


class RuntimeConflict(RuntimeError):
    pass


def claim(family: str) -> None:
    """Assert that no library conflicting with `family` is already loaded in this process."""
    if family not in CONFLICTS:
        raise ValueError(family)
    if os.environ.get("REELRANK_ALLOW_MIXED_OMP") == "1":
        return
    loaded = {f for f, mod in FAMILIES.items() if mod in sys.modules}
    clash = loaded & CONFLICTS[family]
    if clash:
        raise RuntimeConflict(
            f"{family} cannot share a process with {sorted(clash)}: their bundled OpenMP runtimes crash or "
            "deadlock together. Run this step as its own process (see Makefile)."
        )
    # Thread counts are set before the runtime initialises.
    os.environ.setdefault("KMP_WARNINGS", "0")
