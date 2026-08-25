"""LaTeX source for the static Tucker Cloud.ru benchmark. Compile with xelatex."""
import sys
from pathlib import Path


OUT = Path(sys.argv[1])
BATCHES = [1, 2, 4, 8, 16]
VARIANTS = ["Dense AdamW", "Dense Muon", "Static Tucker (5 streams)"]

CAPACITY = {
    "Dense AdamW": {
        "forward_ms": [133.673, 69.485, 38.449, 37.671, 35.598],
        "backward_ms": [227.038, 108.696, 79.712, 75.657, 71.240],
        "full_step_ms": [364.249, 181.508, 120.864, 116.008, 109.514],
        "optimizer_ms": [1.834, 1.845, 1.845, 1.846, 1.845],
        "peak_gb": [3.267, 4.408, 6.610, 11.021, 19.279],
    },
    "Dense Muon": {
        "forward_ms": [136.564, 68.809, 38.752, 37.725, 36.056],
        "backward_ms": [231.445, 108.840, 79.867, 75.630, 71.735],
        "full_step_ms": [413.018, 219.773, 152.999, 148.450, 143.379],
        "optimizer_ms": [42.787, 40.562, 33.526, 34.269, 34.747],
        "peak_gb": [2.965, 4.100, 6.284, 10.694, 18.976],
    },
    "Static Tucker (5 streams)": {
        "forward_ms": [447.973, 222.405, 116.416, 75.769, 73.332],
        "backward_ms": [868.427, 430.499, 213.168, 134.024, 124.755],
        "full_step_ms": [1909.140, 1237.386, 918.959, 786.563, 775.311],
        "optimizer_ms": [586.684, 582.271, 585.267, 574.778, 574.718],
        "peak_gb": [4.234, 6.490, 10.727, 19.417, 36.262],
    },
}

OLD_SEQUENTIAL_OPTIMIZER = [444.376, 424.360, 398.123, 446.803, 439.373]


def render(rows, columns, corner="Метод"):
    header = f"\\textbf{{{corner}}}" + "".join(
        f" & \\textbf{{{column}}}" for column in columns
    )
    lines = [
        r"\small",
        r"\begin{tabular}{l" + "r" * len(columns) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for label, cells in rows:
        lines.append(label + "".join(f" & {cell}" for cell in cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def capacity_rows(key, digits=1):
    return [
        (variant, [f"{value:.{digits}f}" for value in CAPACITY[variant][key]])
        for variant in VARIANTS
    ]


body = [
    r"\section{Forward, мс}",
    r"\begin{center}",
    render(capacity_rows("forward_ms"), [fr"{batch}$\times${16 // batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\section{Backward, мс}",
    r"\begin{center}",
    render(capacity_rows("backward_ms"), [fr"{batch}$\times${16 // batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\section{Optimizer, мс}",
    r"\begin{center}",
    render(
        capacity_rows("optimizer_ms")
        + [("Tucker sequential (old)", [f"{value:.1f}" for value in OLD_SEQUENTIAL_OPTIMIZER])],
        [fr"{batch}$\times${16 // batch}" for batch in BATCHES],
    ),
    r"\end{center}",
    r"\newpage",
    r"\section{Полный шаг, мс}",
    r"\begin{center}",
    render(capacity_rows("full_step_ms"), [fr"{batch}$\times${16 // batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\section{Peak CUDA allocated memory, ГБ}",
    r"\begin{center}",
    render(capacity_rows("peak_gb", digits=2), [fr"{batch}$\times${16 // batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\section{Production: microbatch 32 $\times$ accumulation 4}",
    r"\begin{center}",
    render(
        [
            ("Dense AdamW", ["274.303", "547.362", "1.750", "824.186", "37.441"]),
            ("Dense Muon", ["276.340", "551.348", "35.009", "863.574", "37.136"]),
            ("Static Tucker (5 streams)", ["557.133", "942.166", "571.316", "2073.215", "71.539"]),
        ],
        ["Forward", "Backward", "Optimizer", "Шаг", "Peak, ГБ"],
    ),
    r"\end{center}",
    r"\section{Runs}",
    r"\begin{center}",
    render(
        [
            ("Commit", [r"\texttt{69e569c}"]),
            ("Capacity", [r"\texttt{lm-mpi-job-0583f75d-0fa8-4f7b-8e5e-18d8fae2f90a}"]),
            ("Production", [r"\texttt{lm-mpi-job-f405573b-c021-46cc-bad7-5103d709b023}"]),
        ],
        ["ID"],
    ),
    r"\end{center}",
]

document = r"""\documentclass[10pt]{article}
\usepackage[a4paper,top=15mm,bottom=15mm,left=18mm,right=18mm]{geometry}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{titlesec}
\setmainfont{DejaVu Sans}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.10}
\titleformat{\section}{\Large\bfseries}{\thesection}{0.6em}{}
\titlespacing{\section}{0pt}{11pt}{5pt}
\begin{document}
\begin{center}
{\LARGE\bfseries Static Tucker: benchmark tables}\\[4pt]
{\large H100 80GB HBM3, BF16, Llama 257M, seq 1024}\\[2pt]
{Capacity: 16,384 tokens/step, columns: microbatch $\times$ accumulation}
\end{center}
\vspace{5pt}
BODY
\end{document}
"""

OUT.write_text(document.replace("BODY", "\n".join(body)))
