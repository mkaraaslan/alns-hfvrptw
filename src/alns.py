"""
ALNS main loop with adaptive operator selection (roulette-wheel) and
simulated-annealing-based acceptance criterion.

Implements the algorithm described in the project report
(`vrptw_final_report.tex`, Section "Description of the Metaheuristic Algorithm").
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .instance import Instance
from .solution import Solution
from .construct import construct_initial
from .operators import DESTROY_OPERATORS, REPAIR_OPERATORS


@dataclass
class ALNSParams:
    iterations: int = 8000
    max_time_seconds: float | None = None   # wall-clock budget; None disables it
    segment_length: int = 100
    reaction_factor: float = 0.1
    cooling_rate: float = 0.99975
    sigma1: float = 33.0   # new global best
    sigma2: float = 9.0    # better than current
    sigma3: float = 13.0   # accepted worse
    eta_min_frac: float = 0.10
    eta_max_frac: float = 0.40
    initial_temperature_factor: float = 0.05  # used only as a fallback
    seed: int = 42


@dataclass
class ALNSStats:
    best_cost_history: List[float] = field(default_factory=list)
    current_cost_history: List[float] = field(default_factory=list)
    iter_time_history: List[float] = field(default_factory=list)
    iterations_done: int = 0
    stop_reason: str = ""
    destroy_usage: Dict[str, int] = field(default_factory=dict)
    repair_usage: Dict[str, int] = field(default_factory=dict)
    destroy_scores_total: Dict[str, float] = field(default_factory=dict)
    repair_scores_total: Dict[str, float] = field(default_factory=dict)
    destroy_weights_history: List[Dict[str, float]] = field(default_factory=list)
    repair_weights_history: List[Dict[str, float]] = field(default_factory=list)
    runtime_seconds: float = 0.0


def _roulette(weights: Dict[str, float], rng: random.Random) -> str:
    items = list(weights.items())
    total = sum(w for _, w in items)
    r = rng.random() * total
    cum = 0.0
    for name, w in items:
        cum += w
        if r <= cum:
            return name
    return items[-1][0]


def run_alns(inst: Instance,
             params: ALNSParams | None = None,
             initial: Solution | None = None,
             verbose: bool = False) -> Tuple[Solution, ALNSStats]:
    params = params or ALNSParams()
    rng = random.Random(params.seed)
    stats = ALNSStats()

    sol = initial.copy() if initial is not None else construct_initial(inst)
    sol.evaluate(inst)
    best = sol.copy()
    best.evaluate(inst)

    # initial weights = 1 for every operator
    destroy_weights: Dict[str, float] = {name: 1.0 for name, _ in DESTROY_OPERATORS}
    repair_weights: Dict[str, float] = {name: 1.0 for name, _ in REPAIR_OPERATORS}
    destroy_fn = {name: fn for name, fn in DESTROY_OPERATORS}
    repair_fn = {name: fn for name, fn in REPAIR_OPERATORS}

    seg_destroy_score: Dict[str, float] = {n: 0.0 for n in destroy_weights}
    seg_destroy_count: Dict[str, int] = {n: 0 for n in destroy_weights}
    seg_repair_score: Dict[str, float] = {n: 0.0 for n in repair_weights}
    seg_repair_count: Dict[str, int] = {n: 0 for n in repair_weights}

    for n in destroy_weights:
        stats.destroy_usage[n] = 0
        stats.destroy_scores_total[n] = 0.0
    for n in repair_weights:
        stats.repair_usage[n] = 0
        stats.repair_scores_total[n] = 0.0

    # initial temperature: calibrated from a short random-walk warm-up so it is
    # robust to the initial-solution quality. We sample a few destroy+repair
    # moves from the starting solution and use the mean POSITIVE delta as the
    # "typical worse-move magnitude"; T0 is set so that such a delta is
    # accepted with probability 0.5.
    sample_deltas: List[float] = []
    sample_rng = random.Random(params.seed + 1)
    eta_min = max(1, int(round(params.eta_min_frac * inst.n)))
    eta_max = max(eta_min, int(round(params.eta_max_frac * inst.n)))
    for _ in range(40):
        d_name = sample_rng.choice(list(destroy_weights.keys()))
        r_name = sample_rng.choice(list(repair_weights.keys()))
        d_fn_s = {n: f for n, f in DESTROY_OPERATORS}[d_name]
        r_fn_s = {n: f for n, f in REPAIR_OPERATORS}[r_name]
        eta_s = sample_rng.randint(eta_min, eta_max)
        try:
            partial_s, removed_s = d_fn_s(sol, eta_s, inst, sample_rng)
        except Exception:
            continue
        if removed_s:
            try:
                r_fn_s(partial_s, removed_s, inst, sample_rng)
            except Exception:
                continue
        partial_s.evaluate(inst)
        d = partial_s.cost - sol.cost
        if d > 0:
            sample_deltas.append(d)
    if sample_deltas:
        typical_delta = sum(sample_deltas) / len(sample_deltas)
    else:
        typical_delta = max(1.0, params.initial_temperature_factor * sol.cost)
    T = -typical_delta / math.log(0.5)

    eta_min = max(1, int(round(params.eta_min_frac * inst.n)))
    eta_max = max(eta_min, int(round(params.eta_max_frac * inst.n)))

    t0 = time.time()
    stop_reason = "iterations"
    last_log_time = t0
    for it in range(1, params.iterations + 1):
        # time-budget check (evaluated at the top of each iteration)
        if params.max_time_seconds is not None:
            if time.time() - t0 >= params.max_time_seconds:
                stop_reason = "time"
                break
        d_name = _roulette(destroy_weights, rng)
        r_name = _roulette(repair_weights, rng)
        d_fn = destroy_fn[d_name]
        r_fn = repair_fn[r_name]

        eta = rng.randint(eta_min, eta_max)
        try:
            partial, removed = d_fn(sol, eta, inst, rng)
        except TypeError:
            partial, removed = d_fn(sol, eta, inst, rng)
        candidate = partial
        if removed:
            r_fn(candidate, removed, inst, rng)
        candidate.evaluate(inst)

        seg_destroy_count[d_name] += 1
        seg_repair_count[r_name] += 1
        stats.destroy_usage[d_name] += 1
        stats.repair_usage[r_name] += 1

        score = 0.0
        accept = False
        if candidate.cost < best.cost - 1e-9:
            best = candidate.copy()
            best.evaluate(inst)
            sol = candidate
            score = params.sigma1
            accept = True
        elif candidate.cost < sol.cost - 1e-9:
            sol = candidate
            score = params.sigma2
            accept = True
        else:
            # SA acceptance for worse moves
            delta = candidate.cost - sol.cost
            if T > 1e-12 and rng.random() < math.exp(-delta / T):
                sol = candidate
                score = params.sigma3
                accept = True

        seg_destroy_score[d_name] += score
        seg_repair_score[r_name] += score
        stats.destroy_scores_total[d_name] += score
        stats.repair_scores_total[r_name] += score
        stats.best_cost_history.append(best.cost)
        stats.current_cost_history.append(sol.cost)

        # cooling
        T *= params.cooling_rate

        # segment update
        if it % params.segment_length == 0:
            for n in destroy_weights:
                if seg_destroy_count[n] > 0:
                    avg = seg_destroy_score[n] / seg_destroy_count[n]
                    destroy_weights[n] = ((1 - params.reaction_factor) * destroy_weights[n]
                                          + params.reaction_factor * avg)
                seg_destroy_score[n] = 0.0
                seg_destroy_count[n] = 0
            for n in repair_weights:
                if seg_repair_count[n] > 0:
                    avg = seg_repair_score[n] / seg_repair_count[n]
                    repair_weights[n] = ((1 - params.reaction_factor) * repair_weights[n]
                                          + params.reaction_factor * avg)
                seg_repair_score[n] = 0.0
                seg_repair_count[n] = 0
            # avoid weights collapsing to 0
            for n in destroy_weights:
                destroy_weights[n] = max(destroy_weights[n], 0.05)
            for n in repair_weights:
                repair_weights[n] = max(repair_weights[n], 0.05)
            stats.destroy_weights_history.append(dict(destroy_weights))
            stats.repair_weights_history.append(dict(repair_weights))
            if verbose:
                print(f"[it={it:6d}] best={best.cost:.2f}  cur={sol.cost:.2f}  T={T:.3f}")

    stats.runtime_seconds = time.time() - t0
    stats.iterations_done = it if stop_reason == "time" else params.iterations
    stats.stop_reason = stop_reason
    return best, stats
