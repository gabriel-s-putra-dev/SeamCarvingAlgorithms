"""
Plot docs/Greedy/results/single_seam_timing.csv (written by greedy_bench).
    pip install matplotlib
    python tools/plot_greedy_timing.py
Also prints the empirical growth exponent (time ~ N^k) of each method.
"""
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join("docs", "Greedy", "results")
# Same colours as tools/plot_graphcut_timing.py so figures match across reports.
STYLE = {
    "Greedy": ("#eb6834", "s"),
    "DP (ref)": ("#2a78d6", "o"),
    "GC Dinic": ("#1baf7a", "^"),
}


def main():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "single_seam_timing.csv"))))
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    for name, (color, marker) in STYLE.items():
        pts = [(int(r["pixels"]), max(float(r["seconds"]), 1e-9))
               for r in rows if r["method"] == name]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=2, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#3d3d3a")
        big = [p for p in pts if p[0] >= 6800]
        if len(big) >= 2:
            (x0, y0), (x1, y1) = big[0], big[-1]
            print(f"{name:10s} time ~ N^{math.log(y1 / y0) / math.log(x1 / x0):.2f}  ({x0} -> {x1} px)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pixels (m × n)")
    ax.set_ylabel("seconds to find one vertical seam (log)")
    ax.set_title("Greedy vs optimal seam finding: time per seam vs image size",
                 loc="left", fontsize=12)
    ax.grid(True, which="major", color="#e4e3dd", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(right=ax.get_xlim()[1] * 4)
    fig.tight_layout()
    out = os.path.join(RESULTS, "single_seam_timing.png")
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
