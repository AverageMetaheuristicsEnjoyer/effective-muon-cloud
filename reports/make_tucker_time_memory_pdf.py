"""LaTeX source for the static Tucker Cloud.ru benchmark. Compile with xelatex."""
import sys
from pathlib import Path


OUT = Path(sys.argv[1])
BATCHES = [1, 2, 4, 8, 16]
VARIANTS = ["Dense AdamW", "Dense Muon", "Static Tucker"]

CAPACITY = {
    "Dense AdamW": {
        "forward_backward_ms": [360.727, 178.151, 118.167, 113.306, 106.843],
        "full_step_ms": [364.249, 181.508, 120.864, 116.008, 109.514],
        "optimizer_ms": [1.834, 1.845, 1.845, 1.846, 1.845],
        "peak_gb": [3.267, 4.408, 6.610, 11.021, 19.279],
    },
    "Dense Muon": {
        "forward_backward_ms": [368.460, 177.777, 118.627, 113.372, 107.786],
        "full_step_ms": [413.018, 219.773, 152.999, 148.450, 143.379],
        "optimizer_ms": [42.787, 40.562, 33.526, 34.269, 34.747],
        "peak_gb": [2.965, 4.100, 6.284, 10.694, 18.976],
    },
    "Static Tucker": {
        "forward_backward_ms": [1318.437, 651.523, 329.870, 209.822, 198.285],
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
    r"\section{Протокол}",
    r"Llama 257M, H100 80GB HBM3, BF16, sequence length 1024. Capacity sweep "
    r"обрабатывает 16,384 токена за optimizer step: microbatch 1/2/4/8/16 и "
    r"accumulation 16/8/4/2/1. Для каждой точки выполнено 3 warmup и 12 "
    r"измеряемых шагов.",
    r"\par\medskip",
    r"Основная метрика --- сумма CUDA Event-времён forward и backward. Она не "
    r"включает gradient clipping, optimizer, QR-retraction и vector transport. "
    r"Dense Muon использует 5 Newton--Schulz steps; Tensorion --- 6. У Tucker "
    r"core и четыре factor direction запускаются в пяти CUDA streams.",
    r"\section{Forward + backward, мс}",
    r"\begin{center}",
    render(capacity_rows("forward_backward_ms"), [f"bs {batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\section{Forward + backward throughput, токенов/с}",
    r"\begin{center}",
    render(
        [
            ("Dense AdamW", ["45,419", "91,967", "138,652", "144,600", "153,346"]),
            ("Dense Muon", ["44,466", "92,160", "138,114", "144,516", "152,005"]),
            ("Static Tucker", ["12,427", "25,147", "49,668", "78,085", "82,629"]),
        ],
        [f"bs {batch}" for batch in BATCHES],
    ),
    r"\end{center}",
    r"Dense AdamW и Dense Muon имеют практически одинаковый forward/backward. "
    r"Tucker медленнее Muon в 3.58/3.67/2.78/1.85/1.84 раза. Следовательно, "
    r"разница находится в factorized contraction и её backward, а не в выборе "
    r"optimizer backend.",
    r"\section{Optimizer section, мс}",
    r"Tucker-метрика включает Tensorion/Riemannian-Muon, coupled LR scaling, "
    r"QR-retraction и vector transport.",
    r"\begin{center}",
    render(
        capacity_rows("optimizer_ms")
        + [("Tucker sequential (old)", [f"{value:.1f}" for value in OLD_SEQUENTIAL_OPTIMIZER])],
        [f"bs {batch}" for batch in BATCHES],
    ),
    r"\end{center}",
    r"Пять CUDA streams замедлили optimizer на 29--47\,\%; основная стоимость "
    r"не сводится к пяти последовательным Muon.",
    r"\newpage",
    r"\section{Полный шаг, мс}",
    r"\begin{center}",
    render(capacity_rows("full_step_ms"), [f"bs {batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\section{Peak CUDA allocated memory, ГБ}",
    r"\begin{center}",
    render(capacity_rows("peak_gb", digits=2), [f"bs {batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\section{Production: microbatch 32 $\times$ accumulation 4}",
    r"\begin{center}",
    render(
        [
            ("Dense AdamW", ["821.571", "1.750", "824.186", "37.441"]),
            ("Dense Muon", ["827.650", "35.009", "863.574", "37.136"]),
            ("Static Tucker", ["1499.104", "571.316", "2073.215", "71.539"]),
        ],
        ["F+B, мс", "Opt, мс", "Шаг, мс", "Peak, ГБ"],
    ),
    r"\end{center}",
    r"Production Tucker forward+backward в 1.81 раза медленнее Dense Muon. "
    r"Параллельный optimizer в 1.21 раза медленнее старого результата 473.725 ms.",
    r"\section{Контроль корректности}",
    r"Все 84 Tucker-слоя работали в режиме \texttt{contract}. Self-test заменял "
    r"\texttt{materialize\_weight} на исключение. Commit \texttt{69e569c}; "
    r"capacity job \texttt{lm-mpi-job-0583f75d-0fa8-4f7b-8e5e-18d8fae2f90a}; "
    r"production job \texttt{lm-mpi-job-f405573b-c021-46cc-bad7-5103d709b023}.",
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
{\LARGE\bfseries Static Tucker: forward/backward и Muon}\\[4pt]
{\large H100 80GB HBM3, BF16, Llama 257M, seq 1024}
\end{center}
\vspace{5pt}
BODY
\end{document}
"""

OUT.write_text(document.replace("BODY", "\n".join(body)))
