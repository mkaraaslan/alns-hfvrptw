"""
Run ALNS on Solomon-derived C20 and R20 with a heterogeneous fleet and soft
time windows. Save results to results/ and plot routes + convergence.

Usage:
    python run_experiments.py
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

# allow `from src...` when run from the project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.instance import load_solomon, FleetConfig
from src.solution import (Solution, route_distance, route_load,
                          route_schedule)
from src.construct import construct_initial
from src.alns import run_alns, ALNSParams


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def fmt_route(r, inst) -> str:
    seq = " -> ".join(["0"] + [str(c) for c in r.customers] + ["0"])
    load = route_load(r, inst)
    dist = route_distance(r, inst)
    starts, late = route_schedule(r, inst)
    return (f"  [{r.vehicle_type}] load={load:5.1f} / Q={inst.fleet.Q(r.vehicle_type):.0f}"
            f"  dist={dist:7.2f}  late={late:7.2f}  | {seq}")


def report(name: str, sol: Solution, stats, inst) -> dict:
    sol.evaluate(inst)
    print(f"\n=== {name} ===")
    print(f"runtime: {stats.runtime_seconds:.2f}s   "
          f"iterations: {len(stats.best_cost_history)}")
    print(f"FINAL cost: {sol.cost:.2f}")
    print(f"  vehicle fixed cost : {sol._vehicle_cost:.2f}")
    print(f"  variable (distance): {sol._distance_cost:.2f}")
    print(f"  total lateness     : {sol._lateness:.2f}  "
          f"(penalty/unit = {inst.lateness_penalty})")
    print(f"  vehicles used      : L={sol.count_used('L')}/{inst.fleet.m_L}, "
          f"S={sol.count_used('S')}/{inst.fleet.m_S}")
    print("Routes:")
    for r in sol.routes:
        print(fmt_route(r, inst))

    print("\nOperator usage:")
    print(f"  destroy:")
    for n, c in stats.destroy_usage.items():
        avg = (stats.destroy_scores_total[n] / c) if c else 0.0
        print(f"    {n:12s}  used={c:5d}  total_score={stats.destroy_scores_total[n]:7.1f}  avg_score={avg:5.2f}")
    print(f"  repair:")
    for n, c in stats.repair_usage.items():
        avg = (stats.repair_scores_total[n] / c) if c else 0.0
        print(f"    {n:12s}  used={c:5d}  total_score={stats.repair_scores_total[n]:7.1f}  avg_score={avg:5.2f}")

    summary = {
        "instance": name,
        "iterations": len(stats.best_cost_history),
        "runtime_seconds": stats.runtime_seconds,
        "final_cost": sol.cost,
        "vehicle_fixed_cost": sol._vehicle_cost,
        "distance_cost": sol._distance_cost,
        "total_lateness": sol._lateness,
        "lateness_penalty_per_unit": inst.lateness_penalty,
        "vehicles_used_L": sol.count_used("L"),
        "vehicles_used_S": sol.count_used("S"),
        "fleet_limits": {"m_L": inst.fleet.m_L, "m_S": inst.fleet.m_S},
        "routes": [
            {
                "type": r.vehicle_type,
                "customers": r.customers,
                "load": route_load(r, inst),
                "distance": route_distance(r, inst),
                "lateness": route_schedule(r, inst)[1],
            }
            for r in sol.routes
        ],
        "destroy_usage": stats.destroy_usage,
        "repair_usage": stats.repair_usage,
        "destroy_scores_total": stats.destroy_scores_total,
        "repair_scores_total": stats.repair_scores_total,
    }
    return summary


def plot_routes(name: str, sol: Solution, inst, save_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot disabled: {e}]")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    # plot customers
    for c in inst.customers:
        if c.idx == 0:
            ax.scatter(c.x, c.y, c="black", s=140, marker="s",
                       edgecolor="white", zorder=3, label="Depot")
            ax.text(c.x + 0.6, c.y + 0.6, "0", fontsize=9, weight="bold")
        else:
            ax.scatter(c.x, c.y, c="lightgray", s=60, edgecolor="black", zorder=2)
            ax.text(c.x + 0.6, c.y + 0.6, str(c.idx), fontsize=8)
    colors_L = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf"]
    colors_S = ["#ff7f0e", "#d62728", "#e377c2", "#bcbd22"]
    iL = iS = 0
    for r in sol.routes:
        seq = [0] + r.customers + [0]
        xs = [inst.customers[i].x for i in seq]
        ys = [inst.customers[i].y for i in seq]
        if r.vehicle_type == "L":
            color = colors_L[iL % len(colors_L)]; iL += 1
            ls, lw = "-", 2.0
        else:
            color = colors_S[iS % len(colors_S)]; iS += 1
            ls, lw = "--", 1.6
        ax.plot(xs, ys, ls, color=color, lw=lw,
                label=f"{r.vehicle_type}: {len(r.customers)}c, "
                      f"load={route_load(r, inst):.0f}")
    ax.set_title(f"{name}  |  cost = {sol.cost:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_convergence(name: str, stats, save_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    # left: log-scale full history (shows initial high cost from lateness penalty)
    axes[0].plot(stats.current_cost_history, color="lightsteelblue", lw=0.6, label="current")
    axes[0].plot(stats.best_cost_history, color="navy", lw=1.5, label="best")
    axes[0].set_yscale("log")
    axes[0].set_title(f"{name} — full history (log scale)")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("cost (log)")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend()
    # right: zoom on best/current after the initial drop, linear scale
    best = stats.best_cost_history
    cur = stats.current_cost_history
    if best:
        ymax = max(best[100:]) if len(best) > 100 else max(best)
        ymin = min(best) * 0.95
        axes[1].plot(cur, color="lightsteelblue", lw=0.5, label="current")
        axes[1].plot(best, color="navy", lw=1.5, label="best")
        axes[1].set_ylim(ymin, ymax * 1.5)
        axes[1].set_title(f"{name} — zoomed (after warm-up)")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("cost")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_one(instance_path: Path, fleet: FleetConfig, params: ALNSParams,
            lateness_penalty: float):
    inst = load_solomon(instance_path, fleet=fleet, lateness_penalty=lateness_penalty)
    name = inst.name
    print(f"\n##############################")
    print(f"# Solving {name}  (n={inst.n})")
    print(f"##############################")
    init = construct_initial(inst)
    init.evaluate(inst)
    print(f"Initial solution cost: {init.cost:.2f}  "
          f"(L={init.count_used('L')}, S={init.count_used('S')}, "
          f"late={init.total_lateness(inst):.2f})")
    best, stats = run_alns(inst, params=params, initial=init, verbose=True)
    summary = report(name, best, stats, inst)
    plot_routes(name, best, inst, RESULTS_DIR / f"{name}_routes.png")
    plot_convergence(name, stats, RESULTS_DIR / f"{name}_convergence.png")
    with open(RESULTS_DIR / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    fleet = FleetConfig(
        Q_L=200.0, F_L=200.0, alpha_L=1.0, m_L=3,
        Q_S=100.0, F_S=100.0, alpha_S=1.5, m_S=4,
    )
    params = ALNSParams(
        iterations=8000,
        segment_length=100,
        reaction_factor=0.1,
        cooling_rate=0.99975,
        seed=42,
    )
    lateness_penalty = 50.0  # cost per unit time of lateness

    summaries = []
    for fn in ["C20.txt", "R20.txt"]:
        s = run_one(ROOT / "data" / fn, fleet, params, lateness_penalty)
        summaries.append(s)

    # combined summary
    print("\n\n========== COMBINED SUMMARY ==========")
    for s in summaries:
        print(f"{s['instance']:6s}  cost={s['final_cost']:8.2f}  "
              f"vehFixed={s['vehicle_fixed_cost']:6.1f}  dist={s['distance_cost']:7.2f}  "
              f"late={s['total_lateness']:6.2f}  "
              f"L={s['vehicles_used_L']}  S={s['vehicles_used_S']}")
    with open(RESULTS_DIR / "combined_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
