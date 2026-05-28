"""
Final ALNS run for HFVRPTW (C20 & R20).

Configuration:
    - Initial solution: type_aware (vehicle-type-aware sequential insertion)
    - Stopping criteria: min(2000 iterations, 60 seconds wall-clock)
    - Fleet: Large (Q=200, F=200, α=1.0, m=3), Small (Q=100, F=100, α=1.5, m=4)
    - Lateness penalty: 50 per unit time
    - Seeds: 42, 7, 123

Output:
    - Console summary
    - results/final_<inst>_routes.png
    - results/final_<inst>_convergence.png
    - results/final_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.instance import load_solomon, FleetConfig, Instance
from src.solution import Solution, route_distance, route_load, route_schedule
from src.construct import construct_type_aware
from src.alns import run_alns, ALNSParams, ALNSStats

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────── configuration ───────────────────────
FLEET = FleetConfig(
    Q_L=200.0, F_L=200.0, alpha_L=1.0, m_L=3,
    Q_S=100.0, F_S=100.0, alpha_S=1.5, m_S=4,
)
LATENESS_PENALTY = 50.0
PARAMS_TEMPLATE = ALNSParams(
    iterations=2000,
    max_time_seconds=60.0,
    segment_length=100,
    reaction_factor=0.1,
    cooling_rate=0.998,
    sigma1=33.0,
    sigma2=9.0,
    sigma3=13.0,
    eta_min_frac=0.10,
    eta_max_frac=0.40,
)
SEEDS = [42, 7, 123]
# ──────────────────────────────────────────────────────────────


def fmt_route(r, inst) -> str:
    seq = " -> ".join(["0"] + [str(c) for c in r.customers] + ["0"])
    load = route_load(r, inst)
    dist = route_distance(r, inst)
    _, late = route_schedule(r, inst)
    return (f"  [{r.vehicle_type}] load={load:5.1f}/{inst.fleet.Q(r.vehicle_type):.0f}"
            f"  dist={dist:7.2f}  late={late:.2f}  | {seq}")


def solve_instance(inst: Instance, seed: int) -> dict:
    """Run ALNS on one instance with one seed."""
    t_start = time.time()
    init = construct_type_aware(inst)
    init.evaluate(inst)
    init_time = time.time() - t_start

    params = ALNSParams(
        iterations=PARAMS_TEMPLATE.iterations,
        max_time_seconds=PARAMS_TEMPLATE.max_time_seconds,
        segment_length=PARAMS_TEMPLATE.segment_length,
        reaction_factor=PARAMS_TEMPLATE.reaction_factor,
        cooling_rate=PARAMS_TEMPLATE.cooling_rate,
        sigma1=PARAMS_TEMPLATE.sigma1,
        sigma2=PARAMS_TEMPLATE.sigma2,
        sigma3=PARAMS_TEMPLATE.sigma3,
        eta_min_frac=PARAMS_TEMPLATE.eta_min_frac,
        eta_max_frac=PARAMS_TEMPLATE.eta_max_frac,
        seed=seed,
    )
    best, stats = run_alns(inst, params=params, initial=init, verbose=False)
    best.evaluate(inst)

    return {
        "instance": inst.name,
        "seed": seed,
        "initial_cost": init.cost,
        "initial_lateness": init.total_lateness(inst),
        "init_time_ms": init_time * 1000,
        "final_cost": best.cost,
        "final_lateness": best.total_lateness(inst),
        "vehicles_L": best.count_used("L"),
        "vehicles_S": best.count_used("S"),
        "iterations_done": stats.iterations_done,
        "stop_reason": stats.stop_reason,
        "runtime_seconds": stats.runtime_seconds,
        "best_cost_history": stats.best_cost_history,
        "routes": [
            {"type": r.vehicle_type, "customers": r.customers,
             "load": route_load(r, inst), "distance": route_distance(r, inst),
             "lateness": route_schedule(r, inst)[1]}
            for r in best.routes
        ],
        "_sol": best,
        "_stats": stats,
    }


def plot_routes(name: str, sol: Solution, inst: Instance, save_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    for c in inst.customers:
        if c.idx == 0:
            ax.scatter(c.x, c.y, c="black", s=160, marker="s",
                       edgecolor="white", zorder=3, label="Depot")
            ax.text(c.x + 0.5, c.y + 0.5, "0", fontsize=9, weight="bold")
        else:
            ax.scatter(c.x, c.y, c="lightgray", s=60, edgecolor="black", zorder=2)
            ax.text(c.x + 0.5, c.y + 0.5, str(c.idx), fontsize=8)
    colors_L = ["#1f77b4", "#2ca02c", "#9467bd"]
    colors_S = ["#ff7f0e", "#d62728", "#e377c2", "#bcbd22"]
    iL = iS = 0
    for r in sol.routes:
        seq = [0] + r.customers + [0]
        xs = [inst.customers[i].x for i in seq]
        ys = [inst.customers[i].y for i in seq]
        if r.vehicle_type == "L":
            color = colors_L[iL % len(colors_L)]; iL += 1
            ls, lw = "-", 2.2
        else:
            color = colors_S[iS % len(colors_S)]; iS += 1
            ls, lw = "--", 1.6
        ax.plot(xs, ys, ls, color=color, lw=lw,
                label=f"{r.vehicle_type}: {len(r.customers)}c, "
                      f"load={route_load(r, inst):.0f}")
    ax.set_title(f"{name}  |  cost = {sol.cost:.2f}", fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_convergence(name: str, runs: list[dict], save_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for run in runs:
        h = run["best_cost_history"]
        axes[0].plot(h, lw=1.2, alpha=0.8, label=f"seed={run['seed']}")
    axes[0].set_yscale("log")
    axes[0].set_title(f"{name} — convergence (log scale)")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("best cost (log)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, which="both")

    # zoomed
    warmup = 20
    for run in runs:
        h = run["best_cost_history"]
        axes[1].plot(range(warmup, len(h)), h[warmup:], lw=1.2, alpha=0.8,
                     label=f"seed={run['seed']}")
    axes[1].set_title(f"{name} — zoomed (iter >= {warmup})")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("best cost")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    all_results = []
    for fname in ["C20.txt", "R20.txt"]:
        inst = load_solomon(ROOT / "data" / fname, fleet=FLEET,
                            lateness_penalty=LATENESS_PENALTY)
        print(f"\n{'='*60}")
        print(f" Instance: {inst.name}  (n={inst.n})")
        print(f" Stopping: {PARAMS_TEMPLATE.iterations} iter OR "
              f"{PARAMS_TEMPLATE.max_time_seconds}s")
        print(f"{'='*60}")

        runs = []
        best_run = None
        for seed in SEEDS:
            r = solve_instance(inst, seed)
            runs.append(r)
            if best_run is None or r["final_cost"] < best_run["final_cost"]:
                best_run = r
            print(f"  seed={seed:4d}  stop={r['stop_reason']:10s}  "
                  f"iter={r['iterations_done']:5d}  "
                  f"time={r['runtime_seconds']:.2f}s  "
                  f"cost={r['final_cost']:.2f}  "
                  f"late={r['final_lateness']:.2f}  "
                  f"L={r['vehicles_L']} S={r['vehicles_S']}")

        print(f"\n  ★ Best (seed={best_run['seed']}): "
              f"cost={best_run['final_cost']:.2f}")
        print(f"    Routes:")
        for rt in best_run["routes"]:
            seq = " -> ".join(["0"] + [str(c) for c in rt["customers"]] + ["0"])
            print(f"      [{rt['type']}] load={rt['load']:.0f}  "
                  f"dist={rt['distance']:.2f}  late={rt['lateness']:.2f}  "
                  f"| {seq}")

        # plots
        plot_routes(inst.name, best_run["_sol"], inst,
                    RESULTS_DIR / f"final_{inst.name}_routes.png")
        plot_convergence(inst.name, runs,
                         RESULTS_DIR / f"final_{inst.name}_convergence.png")

        # save without non-serializable fields
        for r in runs:
            r.pop("_sol", None)
            r.pop("_stats", None)
        all_results.append({
            "instance": inst.name,
            "n": inst.n,
            "stopping_criteria": {
                "max_iterations": PARAMS_TEMPLATE.iterations,
                "max_time_seconds": PARAMS_TEMPLATE.max_time_seconds,
            },
            "fleet": {"Q_L": FLEET.Q_L, "F_L": FLEET.F_L, "alpha_L": FLEET.alpha_L,
                      "m_L": FLEET.m_L, "Q_S": FLEET.Q_S, "F_S": FLEET.F_S,
                      "alpha_S": FLEET.alpha_S, "m_S": FLEET.m_S},
            "lateness_penalty": LATENESS_PENALTY,
            "initial_constructor": "type_aware",
            "runs": runs,
        })

    # combined JSON
    with open(RESULTS_DIR / "final_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*60}")
    print(" COMBINED RESULTS")
    print(f"{'='*60}")
    for res in all_results:
        costs = [r["final_cost"] for r in res["runs"]]
        print(f"  {res['instance']:5s}  "
              f"best={min(costs):.2f}  mean={sum(costs)/len(costs):.2f}  "
              f"worst={max(costs):.2f}")
    print(f"\nSaved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
