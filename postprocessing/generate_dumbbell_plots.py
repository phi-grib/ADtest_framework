"""
Generates 2-panel Horizontal Dumbbell Plots (Raw -> OOF shifts per endpoint)
split by In-domain vs. Out-of-domain regimes for:
  1. RR
  2. % Outside AD
  3. EF
  4. AUC

Endpoints are fixed in alphabetical order (A to Z from top to bottom) across
all plots for consistent visual tracking across figures.

Author: analysis helper for Eric March's AD/QSAR domain-shift study.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = str(REPO_ROOT / "results" / "final_analysis" / "figures" / "ad_full_comparison_with_auc.csv")
OUT_DIR = str(REPO_ROOT / "results" / "final_analysis" / "figures")

STRATEGY_COLORS = {"Raw": "#1f77b4", "OOF": "#ff7f0e"}


def dumbbell_plot(
    df: pd.DataFrame,
    metric: str,
    xlabel: str,
    out_path: str,
    ref_line: float = None,
    xlim: tuple = None,
):
    """Draws a 2-panel horizontal dumbbell plot showing the shift from Raw to OOF thresholds per endpoint."""
    piv = df.pivot(
        index=["Endpoint", "Domain"], columns="Strategy", values=metric
    ).reset_index()

    fig, (ax_id, ax_ood) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    domains = [("In-domain", ax_id), ("Out-of-domain", ax_ood)]

    for domain_name, ax in domains:
        sub = piv[piv["Domain"] == domain_name].copy()
        sub = sub.dropna(subset=["Raw", "OOF"])

        # Sort endpoints alphabetically so 'A' appears at the top (y = N-1)
        # down to 'Z' at the bottom (y = 0)
        sub = sub.sort_values("Endpoint", ascending=False).reset_index(drop=True)

        y_positions = np.arange(len(sub))

        # 1. Draw connecting lines between Raw and OOF
        for i, row in sub.iterrows():
            ax.plot(
                [row["Raw"], row["OOF"]],
                [i, i],
                color="gray",
                linestyle="-",
                linewidth=1.5,
                zorder=1,
                alpha=0.7,
            )

        # 2. Draw Raw points
        ax.scatter(
            sub["Raw"],
            y_positions,
            color=STRATEGY_COLORS["Raw"],
            s=60,
            zorder=2,
            label="Raw",
            edgecolor="black",
            linewidth=0.5,
        )

        # 3. Draw OOF points
        ax.scatter(
            sub["OOF"],
            y_positions,
            color=STRATEGY_COLORS["OOF"],
            s=60,
            zorder=3,
            label="OOF",
            edgecolor="black",
            linewidth=0.5,
        )

        # Formatting per subplot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sub["Endpoint"], fontweight="bold", fontsize=9.5)
        ax.set_title(f"{domain_name}", fontweight="bold", fontsize=11)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        # Reference line (drawn without adding a label to the legend)
        if ref_line is not None:
            ax.axvline(
                ref_line,
                color="gray",
                linestyle="--",
                linewidth=1.2,
            )

        if xlim is not None:
            ax.set_xlim(*xlim)

    # Centered shared X-axis label for both subplots
    fig.supxlabel(xlabel, fontweight="bold", fontsize=11)

    # Legend on the right subplot showing only Raw and OOF strategies
    handles, labels = ax_ood.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_ood.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower right",
        frameon=True,
        title="Strategy",
    )

    plt.suptitle(
        f"Raw vs. OOF: {xlabel}",
        fontweight="bold",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(IN_PATH)

    metrics_to_plot = [
        ("%Outside", "% Outside AD", "supfig2_pctOutside_dumbbell.png"),
        ("RR", "Relative Risk (RR)", "supfig3_RR_dumbbell.png"),
        ("EF_outside", "Enrichment Factor Outside AD (EF)", "supfig4_EF_outside_dumbbell.png"),
        ("AUC_flagcount", "Flag-Count ROC-AUC", "supfig5_AUC_dumbbell.png"),
    ]

    for col, xlabel, fname in metrics_to_plot:
        if col in df.columns:
            dumbbell_plot(df, col, xlabel, os.path.join(OUT_DIR, fname))

    print(
        "Saved dumbbell plots"
    )


if __name__ == "__main__":
    main()