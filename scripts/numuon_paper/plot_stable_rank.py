#!/usr/bin/env python3
"""Plot normalized stable-rank curves from training stable_rank.jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_GROUPS = [
    "ffn_gate",
    "ffn_down",
    "ffn_up",
    "attn_k",
    "attn_q",
    "attn_v",
    "attn_o",
    "all_projections",
]

DISPLAY_NAMES = {
    "ffn_gate": "FFN Gate Projection",
    "ffn_down": "FFN Down Projection",
    "ffn_up": "FFN Up Projection",
    "attn_k": "Attention Key Projection",
    "attn_q": "Attention Query Projection",
    "attn_v": "Attention Value Projection",
    "attn_o": "Attention Output Projection",
    "all_projections": "All Projections",
}


def read_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_run(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must be label=path")
    label, path = value.split("=", 1)
    path = Path(path)
    if path.is_dir():
        path = path / "stable_rank.jsonl"
    if not path.exists():
        raise argparse.ArgumentTypeError(f"stable-rank file not found: {path}")
    return label, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run,
                        help="Run label and path, e.g. adamw=exps/.../stable_rank.jsonl")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--out", type=Path, default=Path("artifacts/numuon_paper/stable_rank.png"))
    parser.add_argument("--csv", type=Path, default=Path("artifacts/numuon_paper/stable_rank.csv"))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    runs = [(label, read_jsonl(path)) for label, path in args.run]
    ncols = 2
    nrows = (len(args.groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)

    csv_rows = ["run,group,iter,normalized_mean,normalized_std,mean,std,count"]
    for ax, group in zip(axes.ravel(), args.groups):
        for label, records in runs:
            xs = []
            ys = []
            yerr = []
            for record in records:
                values = record.get("groups", {}).get(group)
                if values is None:
                    continue
                xs.append(record["iter"])
                ys.append(values["normalized_mean"])
                yerr.append(values["normalized_std"])
                csv_rows.append(
                    f"{label},{group},{record['iter']},{values['normalized_mean']},"
                    f"{values['normalized_std']},{values['mean']},{values['std']},{values['count']}"
                )
            if xs:
                ax.plot(xs, ys, label=label)
                lower = [max(0.0, y - e) for y, e in zip(ys, yerr)]
                upper = [min(1.0, y + e) for y, e in zip(ys, yerr)]
                ax.fill_between(xs, lower, upper, alpha=0.15)
        ax.set_title(DISPLAY_NAMES.get(group, group))
        ax.set_xlabel("Training step")
        ax.set_ylabel("Normalized stable rank")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(args.groups):]:
        ax.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.csv.write_text("\n".join(csv_rows) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
