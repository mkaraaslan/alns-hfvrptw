"""
ALNS Parameter Sensitivity Analysis for HFVRPTW.

Tests the impact of key parameters on solution quality:
  - cooling_rate: {0.995, 0.997, 0.998, 0.999}
  - segment_length (Θ): {50, 100, 200}
  - eta_max_frac (destroy ratio): {0.25, 0.40, 0.55}
  - reaction_factor (ρ): {0.05, 0.10, 0.20}

Base config: type_aware constructor, 2000 iter OR 60s, seed=42.
One parameter varies at a time; others at default.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.instance import load_solomon, FleetConfig
from src.construct import construct_type_aware
from src.alns import run_alns, ALNSParams

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FLEET = FleetConfig(
    Q_L=200.0, F_L=200.0, alpha_L=1.0, m_L=3,
    Q_S=100.0, F_S=100.0, alpha_S=1.5, m_S=4,
)
LATENESS_PENALTY = 50.0
BASE_PARAMS = dict(
    iterations=2000,
    max_time_seconds=60.0,
    segment_length=100,
    reaction_factor=0.1,
    cooling_rate=0.998,
    sigma1=33.0, sigma2=9.0, sigma3=13.0,
    eta_min_frac=0.10, eta_max_frac=0.40,
    seed=42,
)
SEEDS = [42, 7, 123]


def run_config(inst, param_overrides: dict, seeds: list[int]) -> dict:
    results = []
    for s in seeds:
        cfg = {**BASE_PARAMS, **param_overrides, "seed": s}
        params = ALNSParams(**cfg)
        init = construct_type_aware(inst)
        init.evaluate(inst)
        best, stats = run_alns(inst, params=params, initial=init, verbose=False)
        best.evaluate(inst)
        results.append({
            "seed": s,
            "cost": best.cost,
            "lateness": best.total_lateness(inst),
            "iterations": stats.iterations_done,
            "runtime": stats.runtime_seconds,
        })
    costs = [r["cost"] for r in results]
    return {
        "best": min(costs),
        "mean": sum(costs) / len(costs),
        "worst": max(costs),
        "runs": results,
    }


def main():
    experiments = {
        "cooling_rate": [0.995, 0.997, 0.998, 0.999],
        "segment_length": [50, 100, 200],
        "eta_max_frac": [0.25, 0.40, 0.55],
        "reaction_factor": [0.05, 0.10, 0.20],
    }

    all_results = {}
    for fname in ["C20.txt", "R20.txt"]:
        inst = load_solomon(ROOT / "data" / fname, fleet=FLEET,
                            lateness_penalty=LATENESS_PENALTY)
        inst_results = {}
        print(f"\n{'='*60}")
        print(f"  Parameter Tuning — {inst.name} (n={inst.n})")
        print(f"{'='*60}")

        for param_name, values in experiments.items():
            print(f"\n  --- {param_name} ---")
            param_results = {}
            for val in values:
                res = run_config(inst, {param_name: val}, SEEDS)
                param_results[str(val)] = res
                print(f"    {param_name}={val:<6}  "
                      f"best={res['best']:.2f}  mean={res['mean']:.2f}  "
                      f"worst={res['worst']:.2f}")
            inst_results[param_name] = param_results
        all_results[inst.name] = inst_results

    with open(RESULTS_DIR / "parameter_tuning.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for inst_name, inst_results in all_results.items():
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            axes = axes.flatten()
            for idx, (param_name, param_results) in enumerate(inst_results.items()):
                ax = axes[idx]
                xs = list(param_results.keys())
                means = [param_results[x]["mean"] for x in xs]
                bests = [param_results[x]["best"] for x in xs]
                worsts = [param_results[x]["worst"] for x in xs]
                x_pos = range(len(xs))
                ax.bar(x_pos, means, width=0.5, alpha=0.6, color="steelblue",
                       label="mean")
                ax.scatter(x_pos, bests, color="green", zorder=3, s=50,
                           label="best", marker="^")
                ax.scatter(x_pos, worsts, color="red", zorder=3, s=50,
                           label="worst", marker="v")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(xs)
                ax.set_xlabel(param_name)
                ax.set_ylabel("cost")
                ax.set_title(f"{inst_name} — {param_name}")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis="y")
            fig.suptitle(f"Parameter Sensitivity — {inst_name}", fontsize=14)
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / f"tuning_{inst_name}.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"  [plot error: {e}]")

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
