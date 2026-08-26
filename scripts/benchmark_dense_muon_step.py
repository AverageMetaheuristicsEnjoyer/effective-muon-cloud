#!/usr/bin/env python3
"""Benchmark the production-size dense Llama with and without Muon."""

import argparse
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(
    0, str(PROJECT_ROOT / "experiments" / "fused_persistent_tucker")
)

from models.utils import get_model
from third_party.lite.muonlite import MuonLite


def make_config(args):
    return SimpleNamespace(
        model="llama",
        vocab_size=50304,
        sequence_length=args.sequence_length,
        n_layer=12,
        n_embd=1024,
        n_head=8,
        ffn_hidden_size=2816,
        multiple_of=256,
        dropout=0.0,
        rmsnorm_eps=1e-6,
        init_std=0.02,
        dtype="bfloat16",
        device="cuda",
        fp8=False,
        fp8_optim=False,
        liger_kernels=True,
        activation_checkpointing=args.activation_checkpointing,
        label_smoothing=0.0,
        attention_type="standard",
        qkv_clipping=False,
        linear_parameterization=args.model_type,
        tucker_rank=259,
        tucker_ranks=None,
        tucker_attention_ranks=None,
        tucker_gate_up_ranks=None,
        tucker_down_ranks=None,
        tucker_rank_plan=None,
        tucker_equal_params=False,
        tucker_forward_mode="chunked_contract",
        tucker_contract_chunk_size=args.chunk_size,
        tucker_head_contract_chunk_size=args.head_chunk_size,
        tucker_dense_adamw_matrices=True,
        tucker_retract_every_step=False,
        tucker_vector_transport=False,
        tucker_riemannian_muon=False,
        target_parameter_count=(
            257676352 if args.model_type == "tucker" else 257188864
        ),
        target_parameter_tolerance=12312,
        batch_size=args.batch_size,
    )


def make_muon(model, args):
    muon_params = []
    adamw_params = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and not any(
            key in name for key in ("wte", "wpe", "lm_head", "embed", "core_logits")
        ):
            muon_params.append((name, parameter))
        else:
            adamw_params.append((name, parameter))
    optimizer = MuonLite(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=1e-3,
        weight_decay=0.1,
        ns_steps=6,
        muon_theta=0.95,
        adamw_betas=(0.9, 0.99),
        adamw_eps=1e-8,
        total_steps=39250,
        warmup_steps=2000,
        beta1=0.0,
        beta2=0.0,
        chi=1.0,
        chi_adamw=1.0,
        subspace_ratio=0.0,
    )
    return optimizer, sum(p.numel() for _, p in muon_params), sum(
        p.numel() for _, p in adamw_params
    )


def mib(value):
    return value / 1024**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=("dense", "tucker"), default="dense")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--optimizer", choices=("none", "muon"), default="muon")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--head-chunk-size", type=int, default=2048)
    parser.add_argument("--fused-backward", action="store_true")
    parser.add_argument("--online-ce", action="store_true")
    parser.add_argument("--output-mode-tile", type=int, default=64)
    args = parser.parse_args()

    if args.online_ce:
        parser.error(
            "--online-ce is not valid for this target: lm_head is dense nn.Linear."
        )

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = get_model(make_config(args)).cuda().train()
    if not isinstance(model.lm_head, torch.nn.Linear):
        raise RuntimeError(
            "Invalid benchmark architecture: lm_head must remain dense nn.Linear."
        )
    tucker_module_count = sum(
        hasattr(module, "materialize_weight") for module in model.modules()
    )
    expected_tucker_modules = 84 if args.model_type == "tucker" else 0
    if tucker_module_count != expected_tucker_modules:
        raise RuntimeError(
            f"Expected {expected_tucker_modules} internal Tucker modules, got "
            f"{tucker_module_count}."
        )
    expected_parameters = 257676352 if args.model_type == "tucker" else 257188864
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"Expected {expected_parameters} parameters, got {actual_parameters}."
        )
    if args.model_type == "tucker" and args.fused_backward:
        from tucker_fused_ops import install

        install(
            fused_backward=args.fused_backward,
            online_ce=False,
            output_mode_tile=args.output_mode_tile,
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_mib = mib(torch.cuda.memory_allocated())
    optimizer = None
    muon_count = adamw_count = 0
    if args.optimizer == "muon":
        optimizer, muon_count, adamw_count = make_muon(model, args)

    tokens = torch.randint(
        0, model.config.vocab_size,
        (args.batch_size, args.sequence_length),
        device="cuda",
    )
    targets = torch.randint_like(tokens, 0, model.config.vocab_size)

    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, targets)["loss"]
        loss.backward()
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        return float(loss.detach())

    # The first Muon step creates momentum/AdamW states and compiles the dynamic
    # Newton--Schulz kernel. It is intentionally excluded from steady-state time.
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    steady_baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    timings = []
    loss = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)

    peak = torch.cuda.max_memory_allocated()
    state_tensor_bytes = sum(
        value.numel() * value.element_size()
        for state in (optimizer.state.values() if optimizer is not None else ())
        for value in state.values()
        if torch.is_tensor(value)
    )
    print(
        f"model={args.model_type} optimizer={args.optimizer} batch={args.batch_size} "
        f"sequence={args.sequence_length} parameters={parameter_count} "
        f"fused_backward={args.fused_backward} online_ce={args.online_ce} "
        f"output_mode_tile={args.output_mode_tile} "
        f"muon_parameters={muon_count} adamw_parameters={adamw_count} "
        f"median_ms={statistics.median(timings):.3f} "
        f"all_ms={','.join(f'{value:.3f}' for value in timings)} "
        f"model_mib={model_mib:.1f} optimizer_state_mib={mib(state_tensor_bytes):.1f} "
        f"steady_baseline_mib={mib(steady_baseline):.1f} "
        f"peak_mib={mib(peak):.1f} step_delta_mib={mib(peak - steady_baseline):.1f} "
        f"loss={loss:.8f}"
    )


if __name__ == "__main__":
    main()
