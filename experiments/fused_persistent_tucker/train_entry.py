#!/usr/bin/env python3
"""Distributed training entrypoint that installs validated Tucker experiments."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = next(
    candidate
    for candidate in (HERE.parents[1], HERE.parents[2])
    if (candidate / "src" / "main.py").is_file()
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

from tucker_fused_ops import install  # noqa: E402


if os.environ.get("TUCKER_ONLINE_CE", "0") == "1":
    raise RuntimeError(
        "TUCKER_ONLINE_CE=1 is invalid for the target architecture: lm_head is dense."
    )

install(
    fused_backward=os.environ.get("TUCKER_FUSED_BACKWARD", "1") == "1",
    online_ce=False,
    output_mode_tile=int(os.environ.get("TUCKER_ONLINE_CE_OUTPUT_TILE", "64")),
)
runpy.run_path(str(ROOT / "src" / "main.py"), run_name="__main__")
