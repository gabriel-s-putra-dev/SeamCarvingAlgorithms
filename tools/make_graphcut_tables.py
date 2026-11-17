"""
Print the README's measurement tables from the CSVs in docs/GraphCutting/results/,
so every number in the README can be traced to a file written by the C++ programs.

    python tools/make_graphcut_tables.py      (from the repo root, after gc_bench and run_graphcut_memory.py)
"""
import csv
import math
import os

R = os.path.join("docs", "GraphCutting", "results")
MIB = 1048576.0


def load(name):
    with open(os.path.join(R, name)) as f:
        return list(csv.DictReader(f))


def ms(sec):
    sec = float(sec)
    if sec < 1e-6:
        return "< 0.001 ms"
    if sec < 1e-3:
        return f"{sec * 1e3:.3f} ms"
    if sec < 10:
        return f"{sec * 1e3:,.1f} ms".replace(",", " ")
    return f"{sec * 1e3:,.0f} ms".replace(",", " ")


def num(x):
    return f"{int(x):,}".replace(",", " ")


def timing():
    rows = load("single_seam_timing.csv")
    by = {}
    for r in rows:
        by.setdefault(int(r["width"]), {})[r["method"]] = r
    print("| Size | DP | Greedy | GC Dinic | GC Edmonds–Karp | Dinic / DP | EK / Dinic |")
    print("|---|---|---|---|---|---|---|")
    for w in (100, 274, 400, 600, 900):
        m = by[w]
        h, px = m["DP"]["height"], int(m["DP"]["pixels"])
        dp, gr, di = (float(m[k]["seconds"]) for k in ("DP", "Greedy", "GC Dinic"))
        ek = m.get("GC Edmonds-Karp")
        ek_time = ms(ek["seconds"]) if ek else "—"
        ek_ratio = f"{float(ek['seconds']) / di:.1f}×" if ek else "—"
        ratio = f"~{di / dp:,.0f}×".replace(",", " ")
        print(f"| {w}×{h} ({px / 1000:.1f}k px) | {ms(dp)} | {ms(gr)} | {ms(di)} | {ek_time} | {ratio} | {ek_ratio} |")
    print()
    for meth, lo, hi in (("DP", 6800, None), ("GC Dinic", 6800, None), ("GC Edmonds-Karp", 6800, None)):
        pts = sorted((int(r["pixels"]), float(r["seconds"])) for r in rows if r["method"] == meth and int(r["pixels"]) >= lo)
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        print(f"{meth}: N^{math.log(y1 / y0) / math.log(x1 / x0):.2f} ({x0} -> {x1} px)")
    print()


def phases():
    print("| Size | Arcs | Graph | Time | Phases | Augmentations | ms per phase |")
    print("|---|---|---|---|---|---|---|")
    for r in load("phase_scaling.csv"):
        print(f"| {r['width']}×{r['height']} | {num(r['arcs'])} | {float(r['graph_mib']):.1f} MiB | "
              f"{float(r['seconds']):.2f} s | {r['phases']} | {num(r['augmentations'])} | {float(r['ms_per_phase']):.1f} |")
    print()


def memory():
    print("| Image | Nodes V | Arcs | Graph cut (counted) | Team DP (counted) | GC / DP |")
    print("|---|---|---|---|---|---|")
    for r in load("memory_usage.csv"):
        print(f"| {r['width']}×{r['height']} | {num(r['nodes'])} | {num(r['arcs'])} | "
              f"{int(r['gc_bytes']) / MIB:.2f} MiB | {int(r['dp_bytes']) / MIB:.2f} MiB | "
              f"{int(r['gc_bytes']) / int(r['dp_bytes']):.0f}× |")
    print()
    peak = {}
    for r in load("memory_peak.csv"):
        peak.setdefault(int(r["width"]), {"h": r["height"], "px": int(r["pixels"])})[r["method"]] = r

    def cell(r, per_px=True):
        if r is None:
            return "—"
        if r["note"] == "below_resolution":
            return "< 1 MiB"
        d = int(r["delta_bytes"])
        return f"{d / MIB:.2f} MiB ({d / int(r['pixels']):.0f} B/px)" if per_px else f"{d / MIB:.2f} MiB"

    print("| Image | GC Dinic peak | GC EK peak | Team DP peak | GC / DP |")
    print("|---|---|---|---|---|")
    for w in sorted(peak):
        p = peak[w]
        g, e, d = p.get("gc-dinic"), p.get("gc-ek"), p.get("dp")
        if g is None or g["note"] == "below_resolution":
            continue
        ratio = (f"{int(g['delta_bytes']) / int(d['delta_bytes']):.0f}×"
                 if d is not None and d["note"] != "below_resolution" else "—")
        print(f"| {w}×{p['h']} | {cell(g)} | {cell(e, False)} | {cell(d)} | {ratio} |")
    print()


def resize():
    rows = load("full_resize.csv")
    print("| Image | Seams removed | DP | Greedy | GC Dinic | GC per seam | Total cost DP / Greedy / GC |")
    print("|---|---|---|---|---|---|---|")
    for img in ("tower", "synthetic"):
        m = {r["method"]: r for r in rows if r["image"] == img}
        k = int(m["DP"]["seams_removed"])
        print(f"| {img} {m['DP']['size'].replace('x', '×')} | {k} vertical | {float(m['DP']['seconds']):.3f} s | "
              f"{float(m['Greedy']['seconds']):.3f} s | {float(m['GC Dinic']['seconds']):.2f} s | "
              f"{1e3 * float(m['GC Dinic']['seconds']) / k:.0f} ms | "
              f"{num(m['DP']['total_cost'])} / {num(m['Greedy']['total_cost'])} / {num(m['GC Dinic']['total_cost'])} |")
    for r in load("both_directions.csv"):
        print(f"\nboth directions: {r['from']} -> {r['to']}, {r['vertical_seams']} v + {r['horizontal_seams']} h, "
              f"{float(r['seconds']):.1f} s, total cost {num(r['total_cost'])}")
    print()
    for r in load("seam_cost.csv"):
        print(f"first seam {r['image']:9s} {r['method']:16s} cost {r['seam_cost']:>6s} aug {r['augmentations']:>5s} phases {r['phases']}")


if __name__ == "__main__":
    timing()
    phases()
    memory()
    resize()
