#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LAYOUT_LABELS = {
    "balanced4": "Tucker-4",
    "order3_input": "Tucker-3: split input",
    "order3_output": "Tucker-3: split output",
    "order3_paired": "Tucker-3: paired balanced",
}
PROFILE_LABELS = {
    "progressive_133m_rank8": "133M",
    "progressive_225m_rank8": "225M",
}


def tex(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def render_table(rows, columns, align=None) -> str:
    align = align or "ll" + "r" * (len(columns) - 2)
    lines = [
        r"\begin{center}",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{column}}}" for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--platform", default="Cloud.ru")
    args = parser.parse_args()

    input_is_tsv = args.input.suffix == ".tsv"
    if input_is_tsv:
        with args.input.open(newline="") as handle:
            completed = [
                row
                for row in csv.DictReader(handle, delimiter="\t")
                if row.get("status") == "complete"
            ]
    else:
        payload = json.loads(args.input.read_text())
        completed = [
            item for item in payload["results"] if item.get("status") == "complete"
        ]

    def benchmark(item, key):
        if input_is_tsv:
            return item[key.removeprefix("tucker_") if key.startswith("tucker_") else key]
        return item["benchmark"][key]

    points = {
        (
            benchmark(item, "tucker_rank_profile"),
            benchmark(item, "tucker_mode_layout"),
            int(benchmark(item, "microbatch")),
        ): item
        for item in completed
    }
    expected = {
        (profile, layout, microbatch)
        for profile in PROFILE_LABELS
        for layout in LAYOUT_LABELS
        for microbatch in (1, 16)
    }
    missing = sorted(expected - set(points))
    if missing:
        raise ValueError(f"Missing completed benchmark points: {missing}")

    def parameter_value(item, key):
        if input_is_tsv:
            column = {
                "actual_parameters": "parameters",
                "parameter_difference_from_dense": "difference_from_dense",
            }[key]
            return int(item[column])
        return int(item["model"][key])

    def time_value(item, key):
        if input_is_tsv:
            column = "median_ms" if key == "host_total_ms" else key
            return float(item[column])
        return float(item["summary"][key]["median"])

    def memory_value(item, key):
        if input_is_tsv:
            column = {
                "forward_backward_peak_allocated_bytes": "forward_backward_peak_gb",
                "peak_allocated_bytes": "full_peak_gb",
            }[key]
            return float(item[column]) * 1e9
        return float(item["memory"][key])

    def gpu_name(item):
        return item["gpu"] if input_is_tsv else item["gpu"]["name"]

    def environment_value(key):
        return "not recorded" if input_is_tsv else completed[0]["environment"][key]

    def rows_for(getter, digits=1):
        rows = []
        for profile in PROFILE_LABELS:
            for layout in LAYOUT_LABELS:
                values = [getter(points[(profile, layout, batch)]) for batch in (1, 16)]
                rows.append(
                    [
                        PROFILE_LABELS[profile],
                        LAYOUT_LABELS[layout],
                        *(f"{value:.{digits}f}" for value in values),
                    ]
                )
        return rows

    time_metrics = (
        ("Forward, ms", lambda item: time_value(item, "forward_ms")),
        ("Backward, ms", lambda item: time_value(item, "backward_ms")),
        ("Optimizer, ms", lambda item: time_value(item, "optimizer_ms")),
        ("Retraction, ms", lambda item: time_value(item, "retraction_ms")),
        ("Full step, ms", lambda item: time_value(item, "host_total_ms")),
    )
    sections = []
    parameter_rows = []
    for profile in PROFILE_LABELS:
        for layout in LAYOUT_LABELS:
            item = points[(profile, layout, 1)]
            parameter_rows.append(
                [
                    PROFILE_LABELS[profile],
                    LAYOUT_LABELS[layout],
                    f"{parameter_value(item, 'actual_parameters'):,}",
                    f"{parameter_value(item, 'parameter_difference_from_dense'):+,}",
                ]
            )
    sections.extend(
        [
            r"\section{Parameter count}",
            render_table(
                parameter_rows,
                ["Budget", "Layout", "Parameters", "vs dense"],
                "llrr",
            ),
        ]
    )
    for title, getter in time_metrics:
        sections.extend(
            [
                rf"\section{{{title}}}",
                render_table(
                    rows_for(getter),
                    ["Budget", "Layout", r"MB 1$\times$16", r"MB 16$\times$1"],
                    "llrr",
                ),
            ]
        )

    sections.extend(
        [
            r"\section{Forward+backward peak allocated, GB}",
            render_table(
                rows_for(
                    lambda item: memory_value(
                        item, "forward_backward_peak_allocated_bytes"
                    )
                    / 1e9,
                    digits=2,
                ),
                ["Budget", "Layout", r"MB 1$\times$16", r"MB 16$\times$1"],
                "llrr",
            ),
            r"\section{Full-step peak allocated, GB}",
            render_table(
                rows_for(
                    lambda item: memory_value(item, "peak_allocated_bytes") / 1e9,
                    digits=2,
                ),
                ["Budget", "Layout", r"MB 1$\times$16", r"MB 16$\times$1"],
                "llrr",
            ),
        ]
    )

    winner_rows = []
    for profile in PROFILE_LABELS:
        for batch in (1, 16):
            candidates = {
                layout: time_value(points[(profile, layout, batch)], "host_total_ms")
                for layout in LAYOUT_LABELS
            }
            winner = min(
                tuple(layout for layout in LAYOUT_LABELS if layout != "balanced4"),
                key=candidates.get,
            )
            baseline = candidates["balanced4"]
            winner_rows.append(
                [
                    PROFILE_LABELS[profile],
                    f"{batch}x{16 // batch}",
                    LAYOUT_LABELS[winner],
                    f"{candidates[winner]:.1f}",
                    f"{candidates[winner] / baseline:.3f}x",
                ]
            )
    sections.extend(
        [
            r"\section{Fastest order-3 full step}",
            render_table(
                winner_rows,
                ["Budget", "MBxAcc", "Layout", "ms", "time/Tucker-4"],
                "lllrr",
            ),
            r"\section{Run}",
            render_table(
                [
                    ["Commit", rf"\texttt{{{tex(args.commit)}}}"],
                    ["Run", rf"\texttt{{{tex(args.job)}}}"],
                    ["GPU", tex(gpu_name(completed[0]))],
                    ["PyTorch", tex(environment_value("torch"))],
                    ["CUDA", tex(environment_value("cuda"))],
                    ["Samples", "3 warmup + 12 measured"],
                ],
                ["Field", "Value"],
                "ll",
            ),
            r"\footnotesize Every order-3 layout has three trainable factors and one fixed singleton buffer. Paired balanced groups three exact input/output factor pairs, pre-packs a BF16 operator, and reuses saved expansion intermediates in backward. Split-input/output leave one side unsplit. Ratios above 1 mean slower than Tucker-4.",
        ]
    )

    document = r"""\documentclass[10pt]{article}
\usepackage[a4paper,top=14mm,bottom=14mm,left=16mm,right=16mm]{geometry}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{titlesec}
\setmainfont{DejaVu Sans}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.08}
\titleformat{\section}{\large\bfseries}{\thesection}{0.5em}{}
\titlespacing{\section}{0pt}{9pt}{4pt}
\begin{document}
\begin{center}
{\LARGE\bfseries Tucker order-3 vs order-4}\\[3pt]
{\large PLATFORM, H100 80GB, BF16 autocast, Llama geometry, seq 1024}\\[2pt]
{16,384 tokens per optimizer step}
\end{center}
BODY
\end{document}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        document.replace("PLATFORM", tex(args.platform)).replace(
            "BODY", "\n".join(sections)
        )
    )


if __name__ == "__main__":
    main()
