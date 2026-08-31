#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


LAYOUT_LABELS = {
    "balanced4": "Tucker-4",
    "order3_input": "Tucker-3: split input",
    "order3_output": "Tucker-3: split output",
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
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    completed = [
        item for item in payload["results"] if item.get("status") == "complete"
    ]
    points = {
        (
            item["benchmark"]["tucker_rank_profile"],
            item["benchmark"]["tucker_mode_layout"],
            int(item["benchmark"]["microbatch"]),
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
        ("Forward, ms", lambda item: item["summary"]["forward_ms"]["median"]),
        ("Backward, ms", lambda item: item["summary"]["backward_ms"]["median"]),
        ("Optimizer, ms", lambda item: item["summary"]["optimizer_ms"]["median"]),
        ("Retraction, ms", lambda item: item["summary"]["retraction_ms"]["median"]),
        ("Full step, ms", lambda item: item["summary"]["host_total_ms"]["median"]),
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
                    f"{item['model']['actual_parameters']:,}",
                    f"{item['model']['parameter_difference_from_dense']:+,}",
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
                    lambda item: item["memory"][
                        "forward_backward_peak_allocated_bytes"
                    ]
                    / 1e9,
                    digits=2,
                ),
                ["Budget", "Layout", r"MB 1$\times$16", r"MB 16$\times$1"],
                "llrr",
            ),
            r"\section{Full-step peak allocated, GB}",
            render_table(
                rows_for(
                    lambda item: item["memory"]["peak_allocated_bytes"] / 1e9,
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
                layout: points[(profile, layout, batch)]["summary"]["host_total_ms"][
                    "median"
                ]
                for layout in LAYOUT_LABELS
            }
            winner = min(
                ("order3_input", "order3_output"),
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
                ["Budget", "MBxAcc", "Layout", "ms", "vs Tucker-4"],
                "lllrr",
            ),
            r"\section{Run}",
            render_table(
                [
                    ["Commit", rf"\texttt{{{tex(args.commit)}}}"],
                    ["Cloud job", rf"\texttt{{{tex(args.job)}}}"],
                    ["GPU", tex(completed[0]["gpu"]["name"])],
                    ["Samples", "3 warmup + 12 measured"],
                ],
                ["Field", "Value"],
                "ll",
            ),
            r"\footnotesize Order-3 is represented by three trainable factors and one fixed singleton buffer. The unsplit side uses the generic GEMM fallback in the current prototype.",
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
{\LARGE\bfseries Tucker order-3 vs order-4}\[3pt]
{\large Cloud.ru, H100 80GB, BF16 autocast, Llama geometry, seq 1024}\[2pt]
{16,384 tokens per optimizer step}
\end{center}
BODY
\end{document}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document.replace("BODY", "\n".join(sections)))


if __name__ == "__main__":
    main()
