"""
Baseline / alternative heuristics for HFVRPTW comparison.

Implements:
1. Nearest Neighbor (NN) — greedy construction with TW awareness
2. Simulated Annealing (SA) — standalone SA with 2-opt* and relocate moves
3. Clarke-Wright (CW) — TW-simple savings only (no ALNS)

All return evaluated Solutions for fair comparison.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Tuple

from .instance import Instance
from .solution import (Route, Solution, route_distance, route_load,
                       route_schedule, best_insertion)
from .construct import (construct_type_aware, construct_clarke_wright_tw_simple,
                        _assign_types)


# ---------------------------------------------------------------------------
# 1. Nearest Neighbor with TW awareness
# ---------------------------------------------------------------------------

def solve_nearest_neighbor(inst: Instance) -> Tuple[Solution, float]:
    """Nearest-neighbour heuristic: at each step, from the current position,
    insert the closest unvisited customer that does not violate capacity.
    Time-window penalty is tolerated (soft TW) but tie-breaks by lateness."""
    t0 = time.time()
    n = inst.n
    unvisited = set(range(1, n + 1))
    routes: List[Route] = []
    n_used = {"L": 0, "S": 0}

    while unvisited:
        # open a new route
        if n_used["L"] < inst.fleet.m_L:
            vtype = "L"
        elif n_used["S"] < inst.fleet.m_S:
            vtype = "S"
        else:
            vtype = "L"  # overflow
        cap = inst.fleet.Q(vtype)
        route_custs: List[int] = []
        current_load = 0.0
        current_pos = 0
        current_time = 0.0

        while True:
            best_c = None
            best_dist = float("inf")
            best_late = float("inf")
            for c in unvisited:
                dem = inst.customers[c].demand
                if current_load + dem > cap + 1e-6:
                    continue
                d = inst.distance[current_pos][c]
                arr = current_time + inst.travel_time[current_pos][c]
                start = max(arr, inst.customers[c].ready)
                late = max(0.0, start - inst.customers[c].due)
                # prefer: first by lateness (0 is best), then by distance
                if (late, d) < (best_late, best_dist):
                    best_c = c
                    best_dist = d
                    best_late = late
            if best_c is None:
                break
            route_custs.append(best_c)
            unvisited.remove(best_c)
            current_load += inst.customers[best_c].demand
            arr = current_time + inst.travel_time[current_pos][best_c]
            current_time = max(arr, inst.customers[best_c].ready) + inst.customers[best_c].service
            current_pos = best_c

        if route_custs:
            routes.append(Route(vehicle_type=vtype, customers=route_custs))
            n_used[vtype] += 1

    _assign_types(routes, inst)
    sol = Solution(routes=routes)
    sol.evaluate(inst)
    return sol, time.time() - t0


# ---------------------------------------------------------------------------
# 2. Standalone Simulated Annealing (SA)
# ---------------------------------------------------------------------------

def _two_opt_star(sol: Solution, inst: Instance, rng: random.Random) -> Solution:
    """Apply a random 2-opt* move (inter-route segment swap).
    Capacity violations are allowed — cost function penalises lateness anyway."""
    if len(sol.routes) < 2:
        return sol
    new = sol.copy()
    r1_idx, r2_idx = rng.sample(range(len(new.routes)), 2)
    r1 = new.routes[r1_idx]
    r2 = new.routes[r2_idx]
    if not r1.customers or not r2.customers:
        return sol
    i = rng.randint(0, len(r1.customers) - 1)
    j = rng.randint(0, len(r2.customers) - 1)
    tail1 = r1.customers[i:]
    tail2 = r2.customers[j:]
    r1.customers = r1.customers[:i] + tail2
    r2.customers = r2.customers[:j] + tail1
    new.routes = [r for r in new.routes if r.customers]
    new.evaluate(inst)
    return new


def _relocate(sol: Solution, inst: Instance, rng: random.Random) -> Solution:
    """Relocate a random customer to a random position in another route."""
    if len(sol.routes) < 2:
        return sol
    new = sol.copy()
    candidates = [i for i, r in enumerate(new.routes) if len(r.customers) > 0]
    if len(candidates) < 2:
        return sol
    src_idx = rng.choice(candidates)
    dst_candidates = [i for i in range(len(new.routes)) if i != src_idx]
    dst_idx = rng.choice(dst_candidates)
    src = new.routes[src_idx]
    dst = new.routes[dst_idx]
    c_pos = rng.randint(0, len(src.customers) - 1)
    c = src.customers.pop(c_pos)
    ins_pos = rng.randint(0, len(dst.customers))
    dst.customers.insert(ins_pos, c)
    new.routes = [r for r in new.routes if r.customers]
    new.evaluate(inst)
    return new


def _or_opt(sol: Solution, inst: Instance, rng: random.Random) -> Solution:
    """Move a segment of 1-3 customers within the same route."""
    new = sol.copy()
    candidates = [i for i, r in enumerate(new.routes) if len(r.customers) >= 3]
    if not candidates:
        return sol
    r_idx = rng.choice(candidates)
    r = new.routes[r_idx]
    seg_len = rng.randint(1, min(3, len(r.customers) - 1))
    start = rng.randint(0, len(r.customers) - seg_len)
    seg = r.customers[start:start + seg_len]
    remaining = r.customers[:start] + r.customers[start + seg_len:]
    ins_pos = rng.randint(0, len(remaining))
    r.customers = remaining[:ins_pos] + seg + remaining[ins_pos:]
    new.evaluate(inst)
    return new


def _two_opt_intra(sol: Solution, inst: Instance, rng: random.Random) -> Solution:
    """Intra-route 2-opt: reverse a segment within a single route."""
    new = sol.copy()
    candidates = [i for i, r in enumerate(new.routes) if len(r.customers) >= 4]
    if not candidates:
        return sol
    r_idx = rng.choice(candidates)
    r = new.routes[r_idx]
    i, j = sorted(rng.sample(range(len(r.customers)), 2))
    r.customers[i:j+1] = r.customers[i:j+1][::-1]
    new.evaluate(inst)
    return new


def _swap_inter(sol: Solution, inst: Instance, rng: random.Random) -> Solution:
    """Swap one customer between two different routes."""
    if len(sol.routes) < 2:
        return sol
    new = sol.copy()
    r1_idx, r2_idx = rng.sample(range(len(new.routes)), 2)
    r1 = new.routes[r1_idx]
    r2 = new.routes[r2_idx]
    if not r1.customers or not r2.customers:
        return sol
    i = rng.randint(0, len(r1.customers) - 1)
    j = rng.randint(0, len(r2.customers) - 1)
    r1.customers[i], r2.customers[j] = r2.customers[j], r1.customers[i]
    new.evaluate(inst)
    return new


_SA_MOVES = [_two_opt_star, _relocate, _or_opt, _two_opt_intra, _swap_inter]


def solve_sa(inst: Instance, iterations: int = 50000,
             max_time: float = 60.0, cooling_rate: float | None = None,
             seed: int = 42) -> Tuple[Solution, float, dict]:
    """Standalone Simulated Annealing for HFVRPTW.

    Neighbourhood: 2-opt*, relocate, or-opt (equal probability).
    Initial solution: CW-TW-simple (best construction heuristic).
    cooling_rate auto-tuned so T → ~1% of T0 by final iteration.
    """
    t0 = time.time()
    rng = random.Random(seed)
    sol = construct_clarke_wright_tw_simple(inst)
    sol.evaluate(inst)
    best = sol.copy()
    best.evaluate(inst)

    # T0 from warm-up
    deltas = []
    for _ in range(30):
        move = rng.choice(_SA_MOVES)
        cand = move(sol, inst, rng)
        d = cand.cost - sol.cost
        if d > 0:
            deltas.append(d)
    T = (-sum(deltas) / len(deltas) / math.log(0.5)) if deltas else 100.0

    # auto-tune cooling so T_final ≈ 0.01 * T0
    if cooling_rate is None:
        cooling_rate = 0.01 ** (1.0 / max(iterations, 1))

    history = []
    it = 0
    while it < iterations:
        if time.time() - t0 >= max_time:
            break
        move = rng.choice(_SA_MOVES)
        candidate = move(sol, inst, rng)
        delta = candidate.cost - sol.cost
        if delta < -1e-9:
            sol = candidate
            if sol.cost < best.cost - 1e-9:
                best = sol.copy()
                best.evaluate(inst)
        elif T > 1e-12 and rng.random() < math.exp(-delta / T):
            sol = candidate
        T *= cooling_rate
        it += 1
        history.append(best.cost)

    elapsed = time.time() - t0
    stats = {"iterations": it, "runtime": elapsed, "history": history}
    return best, elapsed, stats


# ---------------------------------------------------------------------------
# 3. Clarke-Wright TW-simple only (no ALNS — just construction)
# ---------------------------------------------------------------------------

def solve_cw_only(inst: Instance) -> Tuple[Solution, float]:
    """Return the Clarke-Wright TW-simple solution directly (no search)."""
    t0 = time.time()
    sol = construct_clarke_wright_tw_simple(inst)
    sol.evaluate(inst)
    return sol, time.time() - t0
