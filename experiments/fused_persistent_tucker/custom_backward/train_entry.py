#!/usr/bin/env python3
"""Training entrypoint for the selected custom Tucker backward."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parents[1]))

from experiments.fused_persistent_tucker.custom_backward.integration import (  # noqa: E402
    install_custom_backward,
)
from experiments.fused_persistent_tucker.custom_backward.grouped_muon import (  # noqa: E402
    GroupedSmallFactorMuonLite,
)
from experiments.fused_persistent_tucker.custom_backward.parallel_muon import (  # noqa: E402
    ParallelGroupedMuonLite,
)
from experiments.fused_persistent_tucker.custom_backward.grouped_retraction import (  # noqa: E402
    grouped_retract_tucker_modules_,
)


if os.environ.get("TUCKER_ONLINE_CE", "0") == "1":
    raise RuntimeError("TUCKER_ONLINE_CE is forbidden: target lm_head is Dense")

install_custom_backward(
    cache_policy=os.environ.get("TUCKER_CUSTOM_CACHE_POLICY", "hybrid_gate_up")
)
if os.environ.get("TUCKER_PARALLEL_MUON", "1") == "1":
    import third_party.lite.muonlite as muonlite

    core_microbatch = int(os.environ.get("TUCKER_MUON_CORE_MICROBATCH", "1"))
    parallel_streams = int(os.environ.get("TUCKER_MUON_STREAMS", "2"))

    class ConfiguredParallelGroupedMuonLite(ParallelGroupedMuonLite):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                core_microbatch=core_microbatch,
                parallel_streams=parallel_streams,
                **kwargs,
            )

    muonlite.MuonLite = ConfiguredParallelGroupedMuonLite
elif os.environ.get("TUCKER_GROUPED_SMALL_MUON", "0") == "1":
    import third_party.lite.muonlite as muonlite

    muonlite.MuonLite = GroupedSmallFactorMuonLite
if os.environ.get("TUCKER_GROUPED_RETRACTION", "1") == "1":
    import models.tucker_linear as tucker_linear

    tucker_linear.retract_tucker_modules_ = grouped_retract_tucker_modules_
runpy.run_path(str(ROOT / "src" / "main.py"), run_name="__main__")
