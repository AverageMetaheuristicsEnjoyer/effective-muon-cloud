import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.llama import Llama
from models.layerwise_tucker import LayerwiseTuckerMLP


def relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--liger", action="store_true")
    parser.add_argument("--execution", default="reordered")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--stable-grad-buffers", action="store_true")
    parser.add_argument("--rank-fraction", type=float, choices=(0.25, 0.5), default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(11)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for variant in ("dense", "A", "B"):
        config = SimpleNamespace(
            vocab_size=128, sequence_length=32, n_embd=128, n_head=4, n_layer=args.layers,
            dropout=0.0, init_std=0.02, rmsnorm_eps=1e-5, ffn_hidden_size=352,
            multiple_of=32, dtype="bfloat16", device="cuda", liger_kernels=False,
            liger_bf16_residual=False,
            layerwise_tucker_variant=None if variant == "dense" else variant,
            layerwise_attention_ranks=(int(128 * args.rank_fraction), int(32 * args.rank_fraction), 4),
            layerwise_mlp_ranks=(int(352 * args.rank_fraction), int(128 * args.rank_fraction), 2 if variant == "A" else 3),
            layerwise_execution="reference",
        )
        reference = Llama(config).cuda().train()
        optimized_config = SimpleNamespace(**vars(config))
        optimized_config.layerwise_execution = args.execution
        optimized_config.liger_kernels = args.liger
        optimized = Llama(optimized_config).cuda().train()
        optimized.load_state_dict(reference.state_dict())
        reference_parameters = list(reference.parameters())
        parameter_names = [name for name, _ in reference.named_parameters()]
        optimized_parameters = list(optimized.parameters())
        if args.compile_mode != "none":
            for index, block in enumerate(optimized.transformer.h):
                optimized.transformer.h[index] = torch.compile(
                    block, fullgraph=True, dynamic=False, mode=args.compile_mode,
                )
        optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True) for m in (reference, optimized)]
        if args.stable_grad_buffers:
            for parameter in reference_parameters + optimized_parameters:
                parameter.grad = torch.zeros_like(parameter)
        x = torch.randint(128, (2, 32), device="cuda")
        targets = torch.randint_like(x, 128)
        for step in range(3):
            losses = []
            for model in (reference, optimized):
                micro_losses = []
                for _ in range(args.accumulation):
                    if args.compile_mode != "none":
                        torch.compiler.cudagraph_mark_step_begin()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(x, targets)["loss"] / args.accumulation
                    loss.backward()
                    micro_losses.append(loss.detach().clone())
                losses.append(sum(micro_losses))
            errors = [relative_error(b.grad, a.grad) for a, b in zip(reference_parameters, optimized_parameters)]
            row = dict(variant=variant, step=step, loss_relative_error=relative_error(losses[1], losses[0]),
                       max_parameter_gradient_relative_error=max(errors))
            row["worst_gradient_parameter"] = parameter_names[errors.index(max(errors))]
            squared_error = sum((b.grad.float() - a.grad.float()).square().sum() for a, b in zip(reference_parameters, optimized_parameters))
            squared_reference = sum(a.grad.float().square().sum() for a in reference_parameters)
            row["global_gradient_relative_error"] = (squared_error / squared_reference.clamp_min(1e-24)).sqrt().item()
            assert row["loss_relative_error"] < 0.01, row
            assert max(errors) < 0.1, row
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=not args.stable_grad_buffers)
            row["max_parameter_relative_error"] = max(relative_error(b, a) for a, b in zip(reference_parameters, optimized_parameters))
            assert row["max_parameter_relative_error"] < 0.02, row
            rows.append(row)
            print(json.dumps(row), flush=True)
        del reference, optimized, optimizers
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(status="complete", args=vars(args), rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
