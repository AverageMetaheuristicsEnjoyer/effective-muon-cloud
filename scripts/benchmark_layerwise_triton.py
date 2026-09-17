import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.layerwise_tucker_triton import gate_up_swiglu, packed_swiglu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(31)
    torch.set_num_threads(4)
    rows = []
    for tokens in (1024, 16384):
        for rank in (256, 512):
            width = 2816
            x = torch.randn(tokens, rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            w = (torch.randn(2 * width, rank, device="cuda", dtype=torch.bfloat16) / math.sqrt(rank)).requires_grad_()
            dy = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16) / math.sqrt(width)

            def native():
                gate, up = F.linear(x, w).chunk(2, dim=-1)
                return F.silu(gate) * up

            functions = dict(native=native, packed=lambda: packed_swiglu(F.linear(x, w)),
                             fused=lambda: gate_up_swiglu(x, w))
            expected = native()
            expected_gradients = torch.autograd.grad(expected, (x, w), dy)
            for name, function in functions.items():
                actual = function()
                gradients = torch.autograd.grad(actual, (x, w), dy)
                error = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()
                errors = [error(actual, expected)] + [error(a, b) for a, b in zip(gradients, expected_gradients)]
                assert max(errors) < 0.02, (tokens, rank, name, errors)
                samples = []
                for step in range(25):
                    x.grad, w.grad = None, None
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    function().backward(dy)
                    end.record()
                    end.synchronize()
                    if step >= 5:
                        samples.append(start.elapsed_time(end))
                row = dict(tokens=tokens, rank=rank, name=name, relative_errors=errors,
                           median_ms=statistics.median(samples), samples_ms=samples)
                rows.append(row)
                print(json.dumps(row), flush=True)
            del expected, expected_gradients, actual, gradients, x, w, dy
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(status="complete", rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
