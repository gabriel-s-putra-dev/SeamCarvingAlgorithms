"""
Memory experiment driver: runs gc_mem once per (method, size) in a fresh
process and writes docs/GraphCutting/results/memory_peak.csv + memory_peak.png.

    g++ -std=c++14 -O2 -I. tests/GraphCuttingMemory.cpp GraphCutting.cpp DynamicProgramming.cpp -o gc_mem `pkg-config --cflags --libs opencv4`
    python tools/run_graphcut_memory.py [path/to/gc_mem]      (from the repo root, ~5 min)

The counted bytes (memory_usage.csv, written by gc_bench) are drawn for
comparison: they cover only the final graph + solver arrays, not the temporary
edge list that exists while the graph is being built.
"""
import csv
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join("docs", "GraphCutting", "results")
RUNS = {
    "dp": [68, 150, 274, 600, 1200, 1920],
    "gc-dinic": [68, 150, 274, 600, 1200, 1920],
    "gc-ek": [68, 150, 274, 400],          # EK gets too slow above this
}
FIELDS = ["method", "width", "height", "pixels", "baseline_bytes", "peak_bytes",
          "delta_bytes", "seconds", "seam_cost", "note"]
# Same colours/markers as plot_graphcut_timing.py so the two charts read together.
STYLE = {
    "dp": ("DP (team)", "#2a78d6", "o"),
    "gc-dinic": ("GC Dinic", "#1baf7a", "^"),
    "gc-ek": ("GC Edmonds-Karp", "#eda100", "v"),
}
MIB = 1048576.0


def run(binary):
    rows = []
    for method, widths in RUNS.items():
        for w in widths:
            out = subprocess.run([binary, method, str(w)], capture_output=True, text=True, check=True)
            row = dict(zip(FIELDS, out.stdout.strip().split(",")))
            rows.append(row)
            d = int(row["delta_bytes"])
            print(f"{method:9s} {row['width']:>5}x{row['height']:<5} peak +{d / MIB:8.2f} MiB "
                  f"({d / int(row['pixels']):6.1f} B/px)  {float(row['seconds']):8.3f} s  {row['note']}")
    with open(os.path.join(RESULTS, "memory_peak.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    return rows


def plot(rows):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    for method, (name, color, marker) in STYLE.items():
        pts = [(int(r["pixels"]), int(r["delta_bytes"]) / MIB) for r in rows
               if r["method"] == method and r["note"] != "below_resolution"]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=2, label=f"{name}: measured peak")
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#3d3d3a")
    counted = os.path.join(RESULTS, "memory_usage.csv")
    if os.path.exists(counted):
        c = list(csv.DictReader(open(counted)))
        x = [float(r["pixels"]) for r in c]
        ax.plot(x, [int(r["gc_bytes"]) / MIB for r in c], color="#8a8983", linestyle="--", linewidth=2,
                label="GC: final graph + solver arrays (counted)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pixels (m × n)")
    ax.set_ylabel("extra memory for one seam, MiB (log)")
    ax.set_title("Peak memory to find one seam vs image size", loc="left", fontsize=12)
    ax.grid(True, which="major", color="#e4e3dd", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(right=ax.get_xlim()[1] * 3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "memory_peak.png")
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    binary = sys.argv[1] if len(sys.argv) > 1 else os.path.join(".", "gc_mem")
    plot(run(binary))
