"""
Compare three initial-solution constructors for HFVRPTW + ALNS:

  1. type_aware    -- vehicle-type-aware sequential insertion (greedy by L)
  2. clarke_wright -- Clarke and Wright (1964) parallel savings + ex-post type assignment
  3. random        -- random insertion order, random type for new routes

For each instance (C20, R20) and constructor, we run ALNS multiple times
with different RNG seeds and report:
  - initial cost,
  - final cost statistics (mean, min, max, std),
  - relative improvement of ALNS over the initial solution,
  - lateness, vehicles used, runtime.

Output:
  - text table on stdout
  - results/comparison.json with raw data
  - results/comparison_<inst>.png with box plots and convergence plots
"""

from __future__ import annotations

import json
import statistics as stats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.instance import load_solomon, FleetConfig
from src.solution import Solution
from src.construct import (construct_type_aware, construct_clarke_wright,
                           construct_clarke_wright_tw_simple,
                           construct_clarke_wright_tw_full,
                           construct_random)
from src.alns import run_alns, ALNSParams


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _construct(name: str, inst, seed: int) -> Solution:
    if name == "type_aware":
        return construct_type_aware(inst)                   # deterministic
    if name == "clarke_wright":
        return construct_clarke_wright(inst)                # deterministic
    if name == "cw_tw_simple":
        return construct_clarke_wright_tw_simple(inst)      # deterministic
    if name == "cw_tw_full":
        return construct_clarke_wright_tw_full(inst)        # deterministic
    if name == "random":
        return construct_random(inst, seed=seed)            # depends on seed
    raise ValueError(name)


def run_block(instance_path: Path, fleet: FleetConfig, lateness_penalty: float,
              constructors: list[str], seeds: list[int], iterations: int):
    inst = load_solomon(instance_path, fleet=fleet, lateness_penalty=lateness_penalty)
    name = inst.name
    block = {"instance": name, "n": inst.n, "constructors": {}, "history": {}}
    for c in constructors:
        runs = []
        histories = []
        for s in seeds:
            t_init0 = time.time()
            init = _construct(c, inst, seed=s)
            init.evaluate(inst)
            init_time = time.time() - t_init0
            params = ALNSParams(iterations=iterations, seed=s)
            t0 = time.time()
            best, statsObj = run_alns(inst, params=params, initial=init,
                                       verbose=False)
            elapsed = time.time() - t0
            best.evaluate(inst)
            runs.append({
                "seed": s,
                "initial_cost": init.cost,
                "initial_lateness": init.total_lateness(inst),
                "init_time_seconds": init_time,
                "final_cost": best.cost,
                "final_lateness": best.total_lateness(inst),
                "final_vehicles_L": best.count_used("L"),
                "final_vehicles_S": best.count_used("S"),
                "improvement": init.cost - best.cost,
                "rel_improvement": (init.cost - best.cost) / init.cost if init.cost > 0 else 0.0,
                "runtime_seconds": elapsed,
            })
            histories.append(statsObj.best_cost_history)
        block["constructors"][c] = runs
        block["history"][c] = histories  # list of best-cost arrays per seed
    return block


def summarize(block):
    rows = []
    for c, runs in block["constructors"].items():
        init = [r["initial_cost"] for r in runs]
        init_late = [r["initial_lateness"] for r in runs]
        init_rt = [r["init_time_seconds"] for r in runs]
        fin = [r["final_cost"] for r in runs]
        late = [r["final_lateness"] for r in runs]
        rt = [r["runtime_seconds"] for r in runs]
        rel = [r["rel_improvement"] * 100 for r in runs]
        rows.append({
            "constructor": c,
            "init_mean": stats.mean(init),
            "init_late_mean": stats.mean(init_late),
            "init_time_ms_mean": 1000.0 * stats.mean(init_rt),
            "fin_min": min(fin), "fin_mean": stats.mean(fin),
            "fin_max": max(fin),
            "fin_std": stats.pstdev(fin) if len(fin) > 1 else 0.0,
            "late_mean": stats.mean(late),
            "rel_imp_mean": stats.mean(rel),
            "alns_rt_mean": stats.mean(rt),
        })
    return rows


def print_table(name: str, rows):
    print(f"\n=========== {name} ===========")
    hdr = (f"{'constructor':<16}  {'init_cost':>11}  {'init_late':>10}  "
           f"{'init_t(ms)':>10}  {'fin_min':>8}  {'fin_mean':>9}  "
           f"{'fin_std':>7}  {'rel_imp%':>9}  {'ALNS_rt(s)':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['constructor']:<16}  "
              f"{r['init_mean']:11.1f}  {r['init_late_mean']:10.2f}  "
              f"{r['init_time_ms_mean']:10.2f}  "
              f"{r['fin_min']:8.2f}  {r['fin_mean']:9.2f}  {r['fin_std']:7.2f}  "
              f"{r['rel_imp_mean']:9.2f}  {r['alns_rt_mean']:10.2f}")


def plot_block(block, save_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot disabled: {e}]")
        return
    inst_name = block["instance"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        "type_aware": "navy",
        "clarke_wright": "darkorange",
        "cw_tw_simple": "crimson",
        "cw_tw_full": "purple",
        "random": "darkgreen",
    }

    # left: full history on log scale
    for c, hists in block["history"].items():
        if not hists:
            continue
        L = min(len(h) for h in hists)
        avg = [sum(h[t] for h in hists) / len(hists) for t in range(L)]
        axes[0].plot(avg, label=c, color=colors.get(c, None), lw=1.5)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{inst_name} — mean best cost (log scale)")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("best cost (log)")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend()

    # middle: zoomed (after iter 50) on linear scale
    warmup = 50
    final_low = float("inf")
    final_high = -float("inf")
    for c, hists in block["history"].items():
        if not hists:
            continue
        L = min(len(h) for h in hists)
        avg = [sum(h[t] for h in hists) / len(hists) for t in range(L)]
        zoom_avg = avg[warmup:]
        axes[1].plot(range(warmup, L), zoom_avg, label=c, color=colors.get(c, None), lw=1.5)
        final_low = min(final_low, min(zoom_avg))
        final_high = max(final_high, max(zoom_avg[:200] if len(zoom_avg) > 200 else zoom_avg))
    if final_low < float("inf"):
        # zoom slightly above the highest "early after warm-up" value
        axes[1].set_ylim(final_low * 0.95, final_high * 1.05)
    axes[1].set_title(f"{inst_name} — zoomed (iter >= {warmup})")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("best cost")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # right: box plot of final costs
    order = ["type_aware", "clarke_wright", "cw_tw_simple", "cw_tw_full", "random"]
    labels, data = [], []
    for c in order:
        if c in block["constructors"]:
            labels.append(c)
            data.append([r["final_cost"] for r in block["constructors"][c]])
    bp = axes[2].boxplot(data, tick_labels=labels, patch_artist=True)
    palette = ["#aac4e6", "#ffc78a", "#f5a3a3", "#d4a3e6", "#a8d3a3"]
    for patch, col in zip(bp["boxes"], palette):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    for i, vals in enumerate(data):
        xs = [i + 1] * len(vals)
        axes[2].scatter(xs, vals, color="black", zorder=3, s=20)
    axes[2].set_title(f"{inst_name} — final ALNS cost (per seed)")
    axes[2].set_ylabel("final cost")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    fleet = FleetConfig(
        Q_L=200.0, F_L=200.0, alpha_L=1.0, m_L=3,
        Q_S=100.0, F_S=100.0, alpha_S=1.5, m_S=4,
    )
    lateness_penalty = 50.0
    constructors = ["type_aware", "clarke_wright", "cw_tw_simple", "cw_tw_full", "random"]
    seeds = [42, 7, 123, 2024, 99]
    iterations = 5000

    blocks = []
    for fname in ["C20.txt", "R20.txt"]:
        block = run_block(ROOT / "data" / fname, fleet, lateness_penalty,
                          constructors, seeds, iterations)
        rows = summarize(block)
        block["summary"] = rows
        print_table(block["instance"], rows)
        plot_block(block, RESULTS_DIR / f"comparison_{block['instance']}.png")
        blocks.append(block)

    # save raw data without histories (too large) and full histories separately
    raw = []
    for b in blocks:
        copy = {k: v for k, v in b.items() if k != "history"}
        raw.append(copy)
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(raw, f, indent=2)

    print("\nResults saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
