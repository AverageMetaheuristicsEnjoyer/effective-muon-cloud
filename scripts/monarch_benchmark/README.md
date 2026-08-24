# Large-model training-step benchmark

Complete forward + backward + optimizer steps, measured across three axes:
model size, optimizer, and batch size.

**Models** — five Llama shapes, 257M to 6.89B dense-equivalent parameters.

**Optimizers**

| Variant | Backend | Notes |
|---|---|---|
| `dense_adamw` | `torch.optim.AdamW` | fused; the baseline every ratio is taken against |
| `dense_muon` | `dion.Muon` | five BF16 Newton–Schulz iterations |
| `monarch_muon` | `MonarchMuonOptimizer` | four-block Monarch linears, not a dense model |
| `monarch_muon_iso` | `MonarchMuonOptimizer` | the same, widened until the parameter count matches the dense baseline |
| `galore` | `fira.GaLoreAdamW` | SVD projection, rank = density × hidden size |
| `frugal` | `frugal.BlockAdamW` | blockwise state + SignSGD elsewhere; no projector |
| `apollo` | `apollo.APOLLOAdamW` | random projection, channel-wise scaling |
| `apollo_mini` | `apollo.APOLLOAdamW` | rank 1, tensor-wise scaling |
| `fira` | `fira.FiraAdamW` | GaLore projection plus norm-based scaling |

The memory-efficient variants run on the **dense** model and take their
parameter split from `is_proj_params` in `model.get_parameter_group_specs()`,
mirroring the wiring in `src/optim/optimization.py::get_optimizer`. Only the
decayed `nn.Linear` weights are projected, so embeddings and the LM head keep
full AdamW state — which dominates the small models and becomes negligible at
6.89B.

**Batch size** — microbatch 1, 2, 4, 8, 16 with accumulation traded against it
so every point processes the same 16K tokens per optimizer step at sequence
length 1024. Running out of memory is recorded as a result rather than an
error, and larger batch sizes for that model/optimizer are then skipped.

## Precision and scope

All parameters, gradients, and nonscalar momentum/moment tensors are BF16;
scalar counters may remain FP32. Projection matrices are BF16 and are counted
in `optimizer_state_bytes`, split out as `optimizer_projector_bytes`. Newton–
Schulz also uses BF16. This is a performance and capacity benchmark, not a
numerical-equivalence claim for FP32-state pretraining.

Models run eager. Inputs are preallocated, so data loading is excluded.

## Projection rebuilds

A projection is rebuilt every `--update-proj-gap` steps (default 200), which is
longer than warmup plus the measured window, so the reported median is a clean
steady-state step. The rebuild is then measured in its own window with the gap
forced to 1 and its own peak-memory reset, because the FP32 factorization
workspace never appears in the steady state. The report's amortized column adds
that cost back at its true frequency.

This split matters: at 257M an SVD rebuild costs about 14× a normal step, so
folding it into a twelve-sample median would describe neither state.

## Timing and exclusivity

CUDA events measure forward, backward, and optimizer phases on a dedicated
stream, with no phase-level synchronizations. A device-wide synchronization
closes each step and synchronized host latency is the primary metric.
Optimizer state initialization, the first projection build, and DION kernel
compilation all happen during warmup.

On a shared machine the sweep selects one physical GPU, requires three
consecutive idle and process-free checks, rechecks before each worker, polls
compute-process ownership every 250 ms, and rejects a worker if another compute
PID appears at any check.

On an exclusively scheduled GPU — a one-card cloud job — none of that is
possible or meaningful, so `--exclusive-gpu` skips it, takes the GPU UUID from
the driver through torch, and records `exclusive_gpu: true` so the two kinds of
run stay distinguishable in the report.

## Running it

```bash
# shared machine: wait for a free GPU, then sweep
python -m scripts.monarch_benchmark.run_sweep --python .venv/bin/python

# a subset
python -m scripts.monarch_benchmark.run_sweep \
  --models 257m,834m --variants dense_adamw,galore,fira --microbatches 1,2,4

# exclusively scheduled GPU
python -m scripts.monarch_benchmark.run_sweep --exclusive-gpu
```

The sweep is resumable: a point is reused when its recorded controls match the
requested ones, and `results.json` is rewritten after every point.

Build the report again from completed results:

```bash
python -m scripts.monarch_benchmark.build_report \
  --input benchmark_results/monarch-muon-large/results.json \
  --output reports/memory-efficient-optimizer-benchmark.html
```

### cloud.ru through mlsub

`cloud_run.sh` is the mlsub entry point. It captures everything to the
persistent workspace disk and always exits zero, because a failed mlsub job
shows no logs at all.

```bash
ssh brain_lab mlsub run --repo <public mirror> --branch <branch> \
  --entry scripts/monarch_benchmark/cloud_run.sh --image torch28 --no-pip \
  --gpus 1 --args "--models 257m --variants dense_adamw,galore --microbatches 1,2"
```

Pass `selftest` as the first argument to run the unit tests instead of the
sweep, which is the cheap `--gpus cpu` rehearsal, or `peek` to print the
newest log and the points recorded so far — useful mid-sweep, since `mlsub`
gives no other way into the workspace disk.
