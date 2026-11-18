"""
Charts for the Greedy write-up (docs/Greedy/README.md).
    pip install matplotlib
    python tools/plot_greedy_analysis.py      (run from the repo root)

Reads  docs/Greedy/results/seam_costs.csv  (written by greedy_bench)
       docs/Greedy/results/random_maps.csv (written by greedy_analysis)
Writes docs/Greedy/results/seam_cost_per_step.png
       docs/Greedy/results/random_maps.png
and prints the summary numbers quoted in the README.
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join("docs", "Greedy", "results")
# Same colours as tools/plot_greedy_timing.py / plot_graphcut_timing.py.
STYLE = {
    "Greedy": ("#eb6834", "-"),
    "DP": ("#2a78d6", "-"),
    "GC Dinic": ("#1baf7a", "--"),
}


def tidy(ax):
    ax.grid(True, which="major", color="#e4e3dd", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def seam_cost_per_step():
    costs = defaultdict(list)  # (image, method) -> [cost of seam 1, 2, ...]
    for r in csv.DictReader(open(os.path.join(RESULTS, "seam_costs.csv"))):
        costs[(r["image"], r["method"])].append(int(r["seam_cost"]))
    images = sorted({img for img, _ in costs}, key=lambda s: s != "tower")

    fig, axes = plt.subplots(1, len(images), figsize=(11, 4.2), dpi=150)
    for ax, image in zip(axes, images):
        for method, (color, ls) in STYLE.items():
            ys = costs.get((image, method))
            if not ys:
                continue
            ax.plot(range(1, len(ys) + 1), ys, color=color, linestyle=ls, linewidth=1.8, label=method)
        g, d = costs[(image, "Greedy")], costs[(image, "DP")]
        ratios = [a / b for a, b in zip(g, d) if b > 0]
        worse = sum(a > b for a, b in zip(g, d))
        print(f"{image:10s} total greedy {sum(g):8d}  DP {sum(d):8d}  ratio {sum(g) / sum(d):.2f}x | "
              f"per seam: mean {sum(ratios) / len(ratios):.2f}x, max {max(ratios):.2f}x, "
              f"greedy worse on {worse}/{len(g)} seams")
        ax.set_title(f"{image}: cost of each removed seam", loc="left", fontsize=11)
        ax.set_xlabel("seam number (removal order)")
        ax.set_ylabel("seam cost (sum of backward energy)")
        ax.set_ylim(bottom=0)
        tidy(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS, "seam_cost_per_step.png")
    fig.savefig(out)
    print("saved", out)


def random_maps():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "random_maps.csv"))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    colors = {"uniform": "#eb6834", "smooth": "#7a4fd6"}
    for kind, color in colors.items():
        pts = [r for r in rows if r["map"] == kind]
        x = [int(r["size"]) for r in pts]
        axes[0].plot(x, [float(r["mean_ratio"]) for r in pts], color=color, marker="o", linewidth=2,
                     label=f"{kind} (mean)")
        axes[0].plot(x, [float(r["max_ratio"]) for r in pts], color=color, marker="", linewidth=1,
                     linestyle=":", label=f"{kind} (worst of 300)")
        axes[1].plot(x, [float(r["pct_optimal"]) for r in pts], color=color, marker="o", linewidth=2,
                     label=kind)
    axes[0].axhline(1.0, color="#2a78d6", linewidth=1, linestyle="--")
    axes[0].text(256, 0.97, "optimal", color="#2a78d6", fontsize=8, va="top", ha="right")
    axes[0].set_title("Greedy seam cost / optimal seam cost", loc="left", fontsize=11)
    axes[0].set_ylabel("ratio (1 = optimal)")
    axes[0].set_ylim(bottom=0.9)
    axes[1].set_title("How often greedy finds an optimal seam", loc="left", fontsize=11)
    axes[1].set_ylabel("% of 300 random maps")
    axes[1].set_ylim(-3, 103)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("map size (n × n)")
        ax.set_xticks([int(r["size"]) for r in rows if r["map"] == "uniform"])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        tidy(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS, "random_maps.png")
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    seam_cost_per_step()
    random_maps()
