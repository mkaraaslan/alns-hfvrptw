"""
Benchmark: ALNS vs alternative heuristics for HFVRPTW.

Compares:
  1. ALNS (type_aware init, 2000 iter OR 60s)
  2. Standalone SA (2-opt*, relocate, or-opt; 2000 iter OR 60s)
  3. Nearest Neighbour (construction only)
  4. Clarke-Wright TW-simple (construction only)
  5. Type-aware greedy (construction only)

Metrics: cost, lateness, vehicles used, runtime.
Multiple seeds for stochastic methods.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.instance import load_solomon, FleetConfig
from src.solution import route_load, route_distance, route_schedule
from src.construct import construct_type_aware, construct_clarke_wright_tw_simple
from src.alns import run_alns, ALNSParams
from src.baselines import solve_nearest_neighbor, solve_sa, solve_cw_only

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FLEET = FleetConfig(
    Q_L=200.0, F_L=200.0, alpha_L=1.0, m_L=3,
    Q_S=100.0, F_S=100.0, alpha_S=1.5, m_S=4,
)
LATENESS_PENALTY = 50.0
SEEDS = [42, 7, 123]
ALNS_ITERS = 2000
MAX_TIME = 60.0


def run_alns_multi(inst, seeds):
    runs = []
    for s in seeds:
        init = construct_type_aware(inst)
        init.evaluate(inst)
        params = ALNSParams(iterations=ALNS_ITERS, max_time_seconds=MAX_TIME,
                            cooling_rate=0.998, seed=s)
        best, stats = run_alns(inst, params=params, initial=init, verbose=False)
        best.evaluate(inst)
        runs.append({
            "seed": s, "cost": best.cost,
            "lateness": best.total_lateness(inst),
            "vehicles_L": best.count_used("L"),
            "vehicles_S": best.count_used("S"),
            "iterations": stats.iterations_done,
            "runtime": stats.runtime_seconds,
            "history": stats.best_cost_history,
        })
    return runs


def run_sa_multi(inst, seeds):
    runs = []
    for s in seeds:
        best, elapsed, stats = solve_sa(inst, iterations=50000,
                                         max_time=MAX_TIME, seed=s)
        best.evaluate(inst)
        runs.append({
            "seed": s, "cost": best.cost,
            "lateness": best.total_lateness(inst),
            "vehicles_L": best.count_used("L"),
            "vehicles_S": best.count_used("S"),
            "iterations": stats["iterations"],
            "runtime": elapsed,
            "history": stats["history"],
        })
    return runs


def summarize_runs(runs):
    costs = [r["cost"] for r in runs]
    return {
        "best": min(costs),
        "mean": sum(costs) / len(costs),
        "worst": max(costs),
        "lateness_mean": sum(r["lateness"] for r in runs) / len(runs),
        "runtime_mean": sum(r["runtime"] for r in runs) / len(runs),
    }


def main():
    all_results = []

    for fname in ["C20.txt", "R20.txt"]:
        inst = load_solomon(ROOT / "data" / fname, fleet=FLEET,
                            lateness_penalty=LATENESS_PENALTY)
        print(f"\n{'='*70}")
        print(f"  BENCHMARK — {inst.name} (n={inst.n})")
        print(f"{'='*70}")

        results = {"instance": inst.name, "n": inst.n, "methods": {}}

        # 1. ALNS
        print("  Running ALNS...", end=" ", flush=True)
        alns_runs = run_alns_multi(inst, SEEDS)
        results["methods"]["ALNS"] = {"runs": alns_runs, **summarize_runs(alns_runs)}
        print(f"done — best={results['methods']['ALNS']['best']:.2f}")

        # 2. Standalone SA
        print("  Running SA...", end=" ", flush=True)
        sa_runs = run_sa_multi(inst, SEEDS)
        results["methods"]["SA"] = {"runs": sa_runs, **summarize_runs(sa_runs)}
        print(f"done — best={results['methods']['SA']['best']:.2f}")

        # 3. Nearest Neighbour
        print("  Running NN...", end=" ", flush=True)
        nn_sol, nn_time = solve_nearest_neighbor(inst)
        nn_run = [{"seed": 0, "cost": nn_sol.cost,
                   "lateness": nn_sol.total_lateness(inst),
                   "vehicles_L": nn_sol.count_used("L"),
                   "vehicles_S": nn_sol.count_used("S"),
                   "iterations": 0, "runtime": nn_time}]
        results["methods"]["NN"] = {"runs": nn_run, **summarize_runs(nn_run)}
        print(f"done — cost={nn_sol.cost:.2f}")

        # 4. CW TW-simple
        print("  Running CW-TW...", end=" ", flush=True)
        cw_sol, cw_time = solve_cw_only(inst)
        cw_run = [{"seed": 0, "cost": cw_sol.cost,
                   "lateness": cw_sol.total_lateness(inst),
                   "vehicles_L": cw_sol.count_used("L"),
                   "vehicles_S": cw_sol.count_used("S"),
                   "iterations": 0, "runtime": cw_time}]
        results["methods"]["CW-TW"] = {"runs": cw_run, **summarize_runs(cw_run)}
        print(f"done — cost={cw_sol.cost:.2f}")

        # 5. Type-aware greedy (construction only)
        print("  Running Greedy...", end=" ", flush=True)
        t0 = time.time()
        greedy_sol = construct_type_aware(inst)
        greedy_sol.evaluate(inst)
        greedy_time = time.time() - t0
        greedy_run = [{"seed": 0, "cost": greedy_sol.cost,
                       "lateness": greedy_sol.total_lateness(inst),
                       "vehicles_L": greedy_sol.count_used("L"),
                       "vehicles_S": greedy_sol.count_used("S"),
                       "iterations": 0, "runtime": greedy_time}]
        results["methods"]["Greedy"] = {"runs": greedy_run, **summarize_runs(greedy_run)}
        print(f"done — cost={greedy_sol.cost:.2f}")

        # summary table
        print(f"\n  {'Method':<10} {'Best':>9} {'Mean':>9} {'Worst':>9} "
              f"{'Late':>7} {'Time(s)':>8}")
        print(f"  {'-'*55}")
        for mname in ["ALNS", "SA", "NN", "CW-TW", "Greedy"]:
            m = results["methods"][mname]
            print(f"  {mname:<10} {m['best']:9.2f} {m['mean']:9.2f} "
                  f"{m['worst']:9.2f} {m['lateness_mean']:7.2f} "
                  f"{m['runtime_mean']:8.2f}")

        all_results.append(results)

    # save JSON (without histories for size)
    save_data = []
    for res in all_results:
        res_copy = {"instance": res["instance"], "n": res["n"], "methods": {}}
        for mname, mdata in res["methods"].items():
            runs_no_hist = [{k: v for k, v in r.items() if k != "history"}
                           for r in mdata["runs"]]
            res_copy["methods"][mname] = {
                "runs": runs_no_hist,
                "best": mdata["best"], "mean": mdata["mean"],
                "worst": mdata["worst"],
                "lateness_mean": mdata["lateness_mean"],
                "runtime_mean": mdata["runtime_mean"],
            }
        save_data.append(res_copy)
    with open(RESULTS_DIR / "benchmark.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for res in all_results:
            inst_name = res["instance"]
            methods = ["ALNS", "SA", "CW-TW", "NN", "Greedy"]
            colors_bar = {"ALNS": "#1f77b4", "SA": "#ff7f0e", "CW-TW": "#d62728",
                          "NN": "#2ca02c", "Greedy": "#9467bd"}
            means = {m: res["methods"][m]["mean"] for m in methods}
            bests = {m: res["methods"][m]["best"] for m in methods}

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # (0,0) bar chart — log scale for full range
            x = range(len(methods))
            bars = axes[0, 0].bar(x, [means[m] for m in methods],
                                  color=[colors_bar[m] for m in methods],
                                  alpha=0.7, width=0.6)
            for i, m in enumerate(methods):
                axes[0, 0].text(i, means[m] * 1.1, f"{means[m]:.0f}",
                                ha="center", fontsize=8, rotation=45)
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(methods)
            axes[0, 0].set_yscale("log")
            axes[0, 0].set_ylabel("cost (log scale)")
            axes[0, 0].set_title(f"{inst_name} — All Methods (log scale)")
            axes[0, 0].grid(True, alpha=0.3, axis="y", which="both")

            # (0,1) bar chart — zoomed to metaheuristics only
            zoom_methods = ["ALNS", "SA", "CW-TW"]
            x2 = range(len(zoom_methods))
            axes[0, 1].bar(x2, [means[m] for m in zoom_methods],
                           color=[colors_bar[m] for m in zoom_methods],
                           alpha=0.7, width=0.5)
            axes[0, 1].scatter(x2, [bests[m] for m in zoom_methods],
                               color="black", zorder=3, s=60, marker="*",
                               label="best")
            for i, m in enumerate(zoom_methods):
                axes[0, 1].text(i, means[m] + 2, f"{means[m]:.1f}",
                                ha="center", fontsize=9)
            axes[0, 1].set_xticks(x2)
            axes[0, 1].set_xticklabels(zoom_methods)
            axes[0, 1].set_ylabel("cost")
            axes[0, 1].set_title(f"{inst_name} — ALNS vs SA vs CW-TW (zoomed)")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis="y")

            # (1,0) convergence — log scale
            alns_hist = min(res["methods"]["ALNS"]["runs"],
                            key=lambda r: r["cost"]).get("history", [])
            sa_hist = min(res["methods"]["SA"]["runs"],
                          key=lambda r: r["cost"]).get("history", [])
            if alns_hist:
                axes[1, 0].plot(alns_hist, color="#1f77b4", lw=1.5, label="ALNS")
            if sa_hist:
                axes[1, 0].plot(sa_hist, color="#ff7f0e", lw=1.5, label="SA", alpha=0.7)
            nn_cost = res["methods"]["NN"]["best"]
            cw_cost = res["methods"]["CW-TW"]["best"]
            axes[1, 0].axhline(cw_cost, color="#d62728", ls="--", lw=1,
                               label=f"CW-TW={cw_cost:.0f}")
            axes[1, 0].set_yscale("log")
            axes[1, 0].set_xlabel("iteration")
            axes[1, 0].set_ylabel("best cost (log)")
            axes[1, 0].set_title(f"{inst_name} — Convergence (log scale)")
            axes[1, 0].legend(fontsize=9)
            axes[1, 0].grid(True, alpha=0.3, which="both")

            # (1,1) convergence — zoomed to ALNS final range
            if alns_hist:
                warmup = min(20, len(alns_hist) - 1)
                axes[1, 1].plot(range(warmup, len(alns_hist)),
                                alns_hist[warmup:],
                                color="#1f77b4", lw=1.5, label="ALNS")
                axes[1, 1].axhline(cw_cost, color="#d62728", ls="--", lw=1,
                                   label=f"CW-TW={cw_cost:.0f}")
                axes[1, 1].set_xlabel("iteration")
                axes[1, 1].set_ylabel("best cost")
                axes[1, 1].set_title(f"{inst_name} — ALNS Convergence (zoomed)")
                axes[1, 1].legend(fontsize=9)
                axes[1, 1].grid(True, alpha=0.3)

            fig.suptitle(f"Benchmark — {inst_name}", fontsize=14, y=1.01)
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / f"benchmark_{inst_name}.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"\n  [plot error: {e}]")

    print(f"\n{'='*70}")
    print(f"  All results saved to: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
