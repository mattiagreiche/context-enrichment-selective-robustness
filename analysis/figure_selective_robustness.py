from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/reproduction"
PDF_PATH = OUTPUT_DIR / "selective-robustness-contexts.pdf"
SVG_PATH = OUTPUT_DIR / "selective-robustness-contexts.svg"


CONTEXTS = (
    {
        "key": "minimal",
        "label": "Minimal",
        "color": "#0072B2",
        "marker": "o",
        "filled": False,
    },
    {
        "key": "neutral",
        "label": "Neutral padding",
        "color": "#E69F00",
        "marker": "s",
        "filled": False,
    },
    {
        "key": "generic",
        "label": "Generic social background",
        "color": "#009E73",
        "marker": "D",
        "filled": False,
    },
    {
        "key": "decision",
        "label": "Descriptive grounding",
        "color": "#CC79A7",
        "marker": "o",
        "filled": True,
    },
    {
        "key": "normative",
        "label": "Normative grounding",
        "color": "#7656A8",
        "marker": "^",
        "filled": True,
    },
)


CONTEXT_KEYS = {
    "minimal": "minimal",
    "neutral": "neutral_padded",
    "generic": "generic_background",
    "decision": "decision_grounded",
    "normative": "normative_grounded",
}


def load_estimates() -> dict[str, dict[str, tuple[float, ...]]]:
    published = json.loads((ROOT / "data/paper_results.json").read_text())
    result = {}
    for model in ("Sol", "Terra"):
        saved = published["models"][model]["pooled_ten_call_estimates"]
        result[model] = {}
        for short_key, context in CONTEXT_KEYS.items():
            row = saved[context]
            result[model][short_key] = (
                row["surface_sensitivity_pp"],
                row["surface_sensitivity_ci95_low_pp"],
                row["surface_sensitivity_ci95_high_pp"],
                row["world_response_pp"],
                row["world_response_ci95_low_pp"],
                row["world_response_ci95_high_pp"],
            )
    return result


ESTIMATES = load_estimates()


LABEL_OFFSETS = {
    "Sol": {
        "minimal": (7, -8),
        "neutral": (7, -12),
        "generic": (-7, 7),
        "decision": (7, -13),
        "normative": (7, 7),
    },
    "Terra": {
        "minimal": (7, -9),
        "neutral": (7, -12),
        "generic": (-7, -13),
        "decision": (7, 7),
        "normative": (7, 7),
    },
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def draw_panel(ax: mpl.axes.Axes, model: str, panel: str) -> None:
    for context in CONTEXTS:
        key = context["key"]
        x, xlo, xhi, y, ylo, yhi = ESTIMATES[model][key]
        ax.errorbar(
            x,
            y,
            xerr=[[x - xlo], [xhi - x]],
            yerr=[[y - ylo], [yhi - y]],
            fmt=context["marker"],
            markersize=6.7,
            markerfacecolor=context["color"] if context["filled"] else "white",
            markeredgecolor=context["color"],
            markeredgewidth=1.45,
            ecolor=context["color"],
            elinewidth=0.85,
            capsize=2.1,
            capthick=0.85,
            zorder=3,
        )
        dx, dy = LABEL_OFFSETS[model][key]
        horizontal_alignment = "right" if dx < 0 else "left"
        ax.annotate(
            context["label"],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="center",
            fontsize=7.8,
            color="#222222",
            zorder=4,
        )

    ax.axhline(0, color="#555555", linewidth=0.65, alpha=0.6, zorder=0)
    ax.grid(axis="y", color="#D6D6D6", linewidth=0.55, zorder=0)
    ax.set_xlim(0, 44)
    ax.set_ylim(-23, 103)
    ax.set_xticks((0, 10, 20, 30, 40))
    ax.set_yticks((-20, 0, 20, 40, 60, 80, 100))
    ax.set_title(f"({panel}) {model}", loc="left", fontweight="bold", pad=5)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.7)


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.75),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    draw_panel(axes[0], "Sol", "a")
    draw_panel(axes[1], "Terra", "b")
    axes[0].set_ylabel("Regional - local transfer probability (pp)\nPositive = more transfer in the regional world")
    axes[1].tick_params(labelleft=True)

    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.20, top=0.93, wspace=0.18)
    figure.supxlabel(
        "Mean surface sensitivity (pp)\nSmaller = more stable across surface forms",
        x=(axes[0].get_position().x0 + axes[1].get_position().x1) / 2,
        y=0.018,
        fontsize=9,
    )
    pdf_metadata = {
        "Title": "Selective robustness across context constructions",
        "Author": "SiliSocS selective robustness study",
        "Subject": "Surface sensitivity and regional-local transfer response",
        "Keywords": "selective robustness, LLM, first-action-branch probability",
    }
    svg_metadata = {
        "Title": "Selective robustness across context constructions",
        "Description": "Surface sensitivity and regional-local transfer response",
        "Creator": "SiliSocS selective robustness study",
        "Keywords": "selective robustness, LLM, first-action-branch probability",
    }
    figure.savefig(PDF_PATH, format="pdf", dpi=300, metadata=pdf_metadata)
    figure.savefig(SVG_PATH, format="svg", dpi=300, metadata=svg_metadata)
    plt.close(figure)
    print(PDF_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
