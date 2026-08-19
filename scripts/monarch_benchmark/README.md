# Large-model Monarch-Muon training-step benchmark

This benchmark compares complete forward + backward + optimizer steps for:

- four-block Monarch linears with `MonarchMuonOptimizer`;
- dense linears with fused PyTorch AdamW;
- dense linears with the repository's DION Muon implementation.

The sweep uses five Llama shapes from 257M to 6.89B dense-equivalent
parameters. Every optimizer step processes 1,024 tokens as four
microbatch-1, sequence-256 accumulation passes.

## Precision and scope

All model parameters, gradients, and nonscalar optimizer momentum/moment tensors
use BF16; scalar counters may remain FP32. Newton–Schulz also uses BF16. This
makes the 6.89B dense AdamW point fit on one A100 80 GB,
but it is a performance/capacity benchmark rather than a numerical-equivalence
claim for FP32-state pretraining.

Models run eager, matching the framework's default (`--compile` is opt-in).
Inputs are preallocated and data loading is excluded.

## Timing and exclusivity

CUDA events measure forward, backward, and optimizer phases on a dedicated
stream. No phase-level synchronizations are inserted. A device-wide
synchronization closes each step, and synchronized host latency is the primary
metric. Optimizer state initialization and DION kernel compilation occur during
warmup.

The sweep selects one physical GPU and uses it for every point. It requires
three consecutive idle/process-free checks, rechecks before each worker, polls
compute-process ownership every 250 ms during warmup and measurement, and
rejects a worker if another compute PID appears at any check.

Run the resumable sweep:

```bash
.venv/bin/python -m scripts.monarch_benchmark.run_sweep
```

Build the report again from completed results:

```bash
.venv/bin/python -m scripts.monarch_benchmark.build_report \
  --input benchmark_results/monarch-muon-large/results.json \
  --output reports/monarch-muon-large-benchmark.html
```
