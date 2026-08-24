"""LaTeX source for the static Tucker Cloud.ru benchmark. Compile with xelatex."""
import sys
from pathlib import Path


OUT = Path(sys.argv[1])
BATCHES = [1, 2, 4, 8, 16]
VARIANTS = ["Dense AdamW", "Static Tucker"]

CAPACITY = {
    "Dense AdamW": {
        "accumulation": [16, 8, 4, 2, 1],
        "median_ms": [264.095, 153.137, 120.581, 115.263, 109.955],
        "tokens_per_second": [62038, 106990, 135876, 142144, 149006],
        "peak_gb": [3.267, 4.408, 6.610, 11.021, 19.279],
        "optimizer_ms": [1.845, 1.849, 1.844, 1.843, 1.846],
    },
    "Static Tucker": {
        "accumulation": [16, 8, 4, 2, 1],
        "median_ms": [1675.080, 968.180, 645.391, 659.110, 638.757],
        "tokens_per_second": [9781, 16923, 25396, 24858, 25651],
        "peak_gb": [4.057, 6.325, 10.560, 19.258, 36.086],
        "optimizer_ms": [444.376, 424.360, 398.123, 446.803, 439.373],
    },
}


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


def capacity_rows(key, digits=1, thousands=False):
    rows = []
    for variant in VARIANTS:
        values = CAPACITY[variant][key]
        cells = [
            f"{value:,.0f}" if thousands else f"{value:.{digits}f}"
            for value in values
        ]
        rows.append((variant, cells))
    return rows


body = [
    r"\section{Протокол и контроль корректности}",
    r"Сравниваются плотный Llama с 257,188,864 параметрами и static Tucker с "
    r"257,676,352 параметрами. В capacity sweep на каждом шаге обрабатывается "
    r"16,384 токена: microbatch меняется от 1 до 16, а accumulation --- от 16 "
    r"до 1. Для каждой точки выполнено 3 warmup и 12 измеряемых шагов.",
    r"\par\medskip",
    r"Во всех 84 Tucker-слоях зафиксирован режим \texttt{contract}. Отдельный "
    r"self-test заменяет \texttt{materialize\_weight} на исключение и успешно "
    r"выполняет forward, backward, Tensorion step, QR-retraction и vector transport.",
    r"\section{Полный шаг обучения, мс}",
    r"\begin{center}",
    render(capacity_rows("median_ms"), [f"bs {batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\section{Пропускная способность, токенов/с}",
    r"\begin{center}",
    render(
        capacity_rows("tokens_per_second", thousands=True),
        [f"bs {batch}" for batch in BATCHES],
    ),
    r"\end{center}",
    r"\section{Пиковая CUDA allocated memory, ГБ}",
    r"\begin{center}",
    render(capacity_rows("peak_gb", digits=2), [f"bs {batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\newpage",
    r"\section{Время оптимизатора, мс}",
    r"Для Tucker сюда входят Tensorion/Riemannian-Muon, QR-retraction и перенос "
    r"состояния оптимизатора.",
    r"\begin{center}",
    render(capacity_rows("optimizer_ms"), [f"bs {batch}" for batch in BATCHES]),
    r"\end{center}",
    r"\section{Параметры и постоянная память, ГБ}",
    r"\begin{center}",
    render(
        [
            ("Dense AdamW", ["257.189M", "0.514", "1.029"]),
            ("Static Tucker", ["257.676M", "0.515", "0.723"]),
        ],
        ["Параметры", "Модель", "Optimizer state"],
    ),
    r"\end{center}",
    r"Static Tucker уменьшает optimizer state на 30\,\%, но модель практически "
    r"iso-param с плотной и требует больше activation/workspace memory.",
    r"\section{Production point: microbatch 32 $\times$ accumulation 4}",
    r"На один optimizer step приходится 131,072 токена.",
    r"\begin{center}",
    render(
        [
            ("Dense AdamW", ["823.389", "159,186", "37.441", "1.753"]),
            ("Static Tucker", ["1973.803", "66,406", "71.364", "473.725"]),
        ],
        ["Шаг, мс", "Токенов/с", "Peak, ГБ", "Optimizer, мс"],
    ),
    r"\end{center}",
    r"Production-конфигурация Tucker помещается на H100 80GB с запасом около "
    r"8.6 ГБ, но шаг в 2.40 раза медленнее dense AdamW. В capacity sweep при "
    r"microbatch 16 замедление составляет 5.81 раза, а peak memory --- 1.87 от dense.",
    r"\section{Воспроизводимость}",
    r"GPU: NVIDIA H100 80GB HBM3; BF16 storage; sequence length 1024. Benchmark "
    r"commit: \texttt{3cdf678}.",
    r"\par\smallskip Full sweep: "
    r"\texttt{lm-mpi-job-2fe89f50-a619-450c-a2f2-b09f7ca2c6a2}.",
    r"\par Production point: "
    r"\texttt{lm-mpi-job-a8578193-cca9-46ea-a4c3-464fdb1d5a24}.",
]

document = r"""\documentclass[10pt]{article}
\usepackage[a4paper,top=16mm,bottom=16mm,left=20mm,right=20mm]{geometry}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{titlesec}
\setmainfont{DejaVu Sans}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.12}
\titleformat{\section}{\Large\bfseries}{\thesection}{0.6em}{}
\titlespacing{\section}{0pt}{13pt}{6pt}
\begin{document}
\begin{center}
{\LARGE\bfseries Static Tucker: время и память}\\[4pt]
{\large H100 80GB HBM3, BF16, Llama 257M, seq 1024}
\end{center}
\vspace{6pt}
BODY
\end{document}
"""

OUT.write_text(document.replace("BODY", "\n".join(body)))
