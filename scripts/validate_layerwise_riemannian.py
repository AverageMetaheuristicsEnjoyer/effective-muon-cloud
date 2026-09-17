import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from models.llama import Llama
from optim.layerwise_muon import make_dense_muon
from optim.layerwise_riemannian import (
    layerwise_tucker_specs, make_layerwise_riemannian,
    retract_layerwise_reference, retract_layerwise_grouped,
)


def relative_error(actual, expected):
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-8)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--rank-fraction", type=float, default=0.25)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    rows = []
    for variant in ("dense", "A", "B"):
        d = args.width
        config = SimpleNamespace(
            vocab_size=128, sequence_length=32, n_embd=d, n_head=d // 128, n_layer=2,
            ffn_hidden_size=11 * d // 4, dropout=0.0, init_std=0.02, rmsnorm_eps=1e-5,
            multiple_of=8, layerwise_execution="reordered",
            layerwise_tucker_variant=None if variant == "dense" else variant,
            layerwise_attention_ranks=(int(d * args.rank_fraction), int(128 * args.rank_fraction), 4),
            layerwise_mlp_ranks=(int(11 * d // 4 * args.rank_fraction), int(d * args.rank_fraction), 2 if variant == "A" else 3),
        )
        reference = Llama(config).cuda().train()
        if variant != "dense":
            retract_layerwise_reference(layerwise_tucker_specs(reference))
        grouped = copy.deepcopy(reference)
        if variant == "dense":
            ro, go = make_dense_muon(reference), make_dense_muon(grouped, "grouped")
            rs, gs = [], []
        else:
            ro, rs = make_layerwise_riemannian(reference)
            go, gs = make_layerwise_riemannian(grouped, "grouped")
        tokens = torch.randint(128, (2, 32), device="cuda")
        targets = torch.randint_like(tokens, 128)
        for step in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = reference(tokens, targets)["loss"]
            loss.backward()
            # Identical gradients isolate optimizer/transport parity from
            # BF16 backward divergence between two training trajectories.
            for a, b in zip(reference.parameters(), grouped.parameters()):
                b.grad = a.grad.clone()
            ro.step()
            go.step()
            if rs:
                retract_layerwise_reference(rs, ro)
                retract_layerwise_grouped(gs, go)
            parameter_errors, momentum_errors = [], []
            for a, b in zip(reference.parameters(), grouped.parameters()):
                parameter_errors.append(relative_error(b, a))
                key = "momentum_buffer" if "momentum_buffer" in ro.state[a] else "momentum"
                if key in ro.state[a]:
                    momentum_errors.append(relative_error(go.state[b][key], ro.state[a][key]))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                yr = reference(tokens, targets, get_logits=True)["logits"]
                yg = grouped(tokens, targets, get_logits=True)["logits"]
            row = dict(variant=variant, step=step,
                       max_parameter_relative_error=max(parameter_errors),
                       max_momentum_relative_error=max(momentum_errors),
                       logits_relative_error=relative_error(yg.float(), yr.float()))
            assert row["max_parameter_relative_error"] < 0.003, row
            assert row["max_momentum_relative_error"] < 0.003, row
            assert row["logits_relative_error"] < 0.01, row
            rows.append(row)
            print(json.dumps(row), flush=True)
            ro.zero_grad(set_to_none=True)
            go.zero_grad(set_to_none=True)
        del reference, grouped, ro, go, rs, gs
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(status="complete", args=vars(args), rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
