"""
Destroy and repair operators for ALNS on HFVRPTW (soft time windows).

Destroy operators (return: removed_solution, removed_customer_list):
  1. random_removal
  2. worst_removal
  3. shaw_removal (related removal)
  4. route_removal
  5. vehicle_type_aware_removal

Repair operators (mutate solution to insert all customers from request bank):
  1. greedy_insertion
  2. regret_k_insertion (k=2, k=3)
  3. vehicle_type_aware_insertion
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from .instance import Instance
from .solution import (Route, Solution, best_insertion, insertion_delta,
                       route_distance, route_load, route_schedule)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _all_customer_locations(sol: Solution) -> List[Tuple[int, int, int]]:
    """Return [(route_index, position, customer_id), ...] for every customer in sol."""
    out = []
    for ri, r in enumerate(sol.routes):
        for pi, c in enumerate(r.customers):
            out.append((ri, pi, c))
    return out


def _remove_from_solution(sol: Solution, customer_ids: List[int]) -> None:
    cset = set(customer_ids)
    for r in sol.routes:
        r.customers = [c for c in r.customers if c not in cset]
    # drop empty routes
    sol.routes = [r for r in sol.routes if r.customers]
    sol.invalidate_cache()


# --------------------------------------------------------------------------
# Destroy operators
# --------------------------------------------------------------------------

def random_removal(sol: Solution, eta: int, inst: Instance,
                   rng: random.Random) -> Tuple[Solution, List[int]]:
    new_sol = sol.copy()
    locs = _all_customer_locations(new_sol)
    if not locs:
        return new_sol, []
    eta = min(eta, len(locs))
    chosen = rng.sample(locs, eta)
    removed = [c for (_, _, c) in chosen]
    _remove_from_solution(new_sol, removed)
    return new_sol, removed


def worst_removal(sol: Solution, eta: int, inst: Instance,
                  rng: random.Random, p_worst: float = 3.0) -> Tuple[Solution, List[int]]:
    new_sol = sol.copy()
    new_sol.evaluate(inst)
    removed: List[int] = []
    for _ in range(eta):
        # rank customers by removal gain (higher = more "worst" / expensive)
        contribs: List[Tuple[float, int]] = []
        # compute current solution cost
        cur_cost = new_sol.evaluate(inst)
        for ri, r in enumerate(new_sol.routes):
            for pi, c in enumerate(r.customers):
                # cost without this customer
                tmp_route = Route(vehicle_type=r.vehicle_type,
                                  customers=r.customers[:pi] + r.customers[pi+1:])
                # build a cost delta from old cost using local evaluation
                fleet = inst.fleet
                v = r.vehicle_type
                old_dist = route_distance(r, inst)
                _, old_late = route_schedule(r, inst)
                if tmp_route.customers:
                    new_dist = route_distance(tmp_route, inst)
                    _, new_late = route_schedule(tmp_route, inst)
                    delta = (fleet.alpha(v) * (new_dist - old_dist)
                             + inst.lateness_penalty * (new_late - old_late))
                else:
                    # removing the last customer drops the route entirely (saves F_v)
                    delta = -(fleet.F(v) + fleet.alpha(v) * old_dist
                              + inst.lateness_penalty * old_late)
                gain = -delta  # gain from removing == reduction in cost
                contribs.append((gain, c))
        if not contribs:
            break
        contribs.sort(key=lambda x: -x[0])  # descending gain
        L = len(contribs)
        idx = int(math.floor((rng.random() ** p_worst) * L))
        idx = min(idx, L - 1)
        _, c = contribs[idx]
        _remove_from_solution(new_sol, [c])
        removed.append(c)
    return new_sol, removed


def shaw_removal(sol: Solution, eta: int, inst: Instance,
                 rng: random.Random,
                 phi1: float = 9.0, phi2: float = 3.0, phi3: float = 2.0,
                 p: float = 6.0) -> Tuple[Solution, List[int]]:
    """Related (Shaw) removal: iteratively remove customers similar to one
    previously removed. Similarity is normalized over distance, time-window
    midpoint, and demand.
    """
    new_sol = sol.copy()
    locs = _all_customer_locations(new_sol)
    if not locs:
        return new_sol, []
    customers_in_sol = [c for (_, _, c) in locs]
    # normalization constants
    n = len(inst.customers)
    d_max = max((max(row) for row in inst.distance), default=1.0) or 1.0
    horizon = inst.depot.due if inst.depot.due > 0 else 1.0
    q_max = max((c.demand for c in inst.customers[1:]), default=1.0) or 1.0

    def relatedness(i: int, j: int) -> float:
        ci, cj = inst.customers[i], inst.customers[j]
        d = inst.distance[i][j] / d_max
        t = (abs(ci.ready - cj.ready) + abs(ci.due - cj.due)) / (2.0 * horizon)
        q = abs(ci.demand - cj.demand) / q_max
        return phi1 * d + phi2 * t + phi3 * q

    removed: List[int] = []
    seed = rng.choice(customers_in_sol)
    removed.append(seed)
    _remove_from_solution(new_sol, [seed])
    while len(removed) < eta:
        remaining = [c for c in customers_in_sol if c not in set(removed)]
        if not remaining:
            break
        ref = rng.choice(removed)
        # rank remaining by relatedness to ref (smaller = more related)
        ranked = sorted(remaining, key=lambda j: relatedness(ref, j))
        idx = int(math.floor((rng.random() ** p) * len(ranked)))
        idx = min(idx, len(ranked) - 1)
        c = ranked[idx]
        _remove_from_solution(new_sol, [c])
        removed.append(c)
    return new_sol, removed


def route_removal(sol: Solution, eta: int, inst: Instance,
                  rng: random.Random) -> Tuple[Solution, List[int]]:
    new_sol = sol.copy()
    if not new_sol.routes:
        return new_sol, []
    # remove one or more entire routes until eta customers (approximately) are gone
    routes = new_sol.routes[:]
    rng.shuffle(routes)
    removed: List[int] = []
    for r in routes:
        if len(removed) >= eta:
            break
        removed.extend(r.customers)
    _remove_from_solution(new_sol, removed)
    return new_sol, removed


def vehicle_type_aware_removal(sol: Solution, eta: int, inst: Instance,
                                rng: random.Random) -> Tuple[Solution, List[int]]:
    """Choose a vehicle type with probability proportional to its average
    per-customer cost in the current solution, then remove eta customers
    drawn (uniformly) from routes of that type.
    """
    new_sol = sol.copy()
    type_costs = {"L": 0.0, "S": 0.0}
    type_counts = {"L": 0, "S": 0}
    for r in new_sol.routes:
        if not r.customers:
            continue
        v = r.vehicle_type
        d = route_distance(r, inst)
        cost = inst.fleet.F(v) + inst.fleet.alpha(v) * d
        type_costs[v] += cost
        type_counts[v] += len(r.customers)
    avg_cost = {}
    for v in ("L", "S"):
        avg_cost[v] = type_costs[v] / type_counts[v] if type_counts[v] else 0.0
    s = avg_cost["L"] + avg_cost["S"]
    if s <= 0.0:
        v_pick = rng.choice(["L", "S"])
    else:
        v_pick = "L" if rng.random() < avg_cost["L"] / s else "S"
    pool = []
    for r in new_sol.routes:
        if r.vehicle_type == v_pick:
            pool.extend(r.customers)
    if not pool:
        # fall back to random
        return random_removal(sol, eta, inst, rng)
    eta = min(eta, len(pool))
    removed = rng.sample(pool, eta)
    _remove_from_solution(new_sol, removed)
    return new_sol, removed


DESTROY_OPERATORS = [
    ("random", random_removal),
    ("worst", worst_removal),
    ("shaw", shaw_removal),
    ("route", route_removal),
    ("type_aware", vehicle_type_aware_removal),
]


# --------------------------------------------------------------------------
# Repair operators
# --------------------------------------------------------------------------

def _try_open_new_route(sol: Solution, customer: int, inst: Instance) -> Tuple[float, str] | None:
    """Try opening a new route for a single customer. Returns (delta, type) or None."""
    cdem = inst.customers[customer].demand
    best = None
    fleet = inst.fleet
    n_used = {"L": sol.count_used("L"), "S": sol.count_used("S")}
    # prefer Large if room and capacity OK; otherwise Small.
    for v in ("L", "S"):
        if n_used[v] >= fleet.m(v):
            continue
        if cdem > fleet.Q(v) + 1e-6:
            continue
        # cost = F_v + alpha_v * 2*dist(0,c) + lateness_penalty * lateness
        d_out = inst.distance[0][customer]
        d_back = inst.distance[customer][0]
        # service start at customer:
        cust = inst.customers[customer]
        arrival = d_out
        if arrival < cust.ready:
            arrival = cust.ready
        late = max(0.0, arrival - cust.due)
        # depot return:
        dep_ret = arrival + cust.service + d_back
        late += max(0.0, dep_ret - inst.depot.due)
        delta = fleet.F(v) + fleet.alpha(v) * (d_out + d_back) + inst.lateness_penalty * late
        if best is None or delta < best[0]:
            best = (delta, v)
    return best


def greedy_insertion(sol: Solution, removed: List[int], inst: Instance,
                     rng: random.Random) -> None:
    bank = removed[:]
    while bank:
        best = (float("inf"), -1, -1, -1)  # (delta, cust, route_idx, position) -- route_idx=-1 => new route
        best_new_type = None
        for c in bank:
            # try existing routes
            for ri, r in enumerate(sol.routes):
                d, pos = best_insertion(r, c, inst)
                if d < best[0]:
                    best = (d, c, ri, pos)
                    best_new_type = None
            # try opening a new route
            new = _try_open_new_route(sol, c, inst)
            if new is not None and new[0] < best[0]:
                best = (new[0], c, -1, -1)
                best_new_type = new[1]
        delta, c, ri, pos = best
        if c == -1:
            # capacity infeasible everywhere AND no new vehicle -> drop into nearest large/small with capacity
            c = bank[0]
            # last-resort: pick any feasible existing route by capacity
            placed = False
            for r in sol.routes:
                if route_load(r, inst) + inst.customers[c].demand <= inst.fleet.Q(r.vehicle_type) + 1e-6:
                    r.customers.append(c)
                    placed = True
                    break
            bank.remove(c)
            if not placed:
                # nothing we can do -> abort (caller should retry)
                continue
        else:
            if ri == -1:
                # open new route
                new_route = Route(vehicle_type=best_new_type, customers=[c])
                sol.routes.append(new_route)
            else:
                sol.routes[ri].customers.insert(pos, c)
            bank.remove(c)
    sol.invalidate_cache()


def regret_k_insertion(sol: Solution, removed: List[int], inst: Instance,
                       rng: random.Random, k: int = 2) -> None:
    bank = removed[:]
    while bank:
        # for each c in bank, compute best k positions across distinct routes
        best_overall = (-float("inf"),)  # max regret
        chosen = None
        for c in bank:
            # collect (delta, ri or 'new', pos, new_type)
            options: List[Tuple[float, int, int, str | None]] = []
            for ri, r in enumerate(sol.routes):
                d, pos = best_insertion(r, c, inst)
                if pos != -1 and d < float("inf"):
                    options.append((d, ri, pos, None))
            new = _try_open_new_route(sol, c, inst)
            if new is not None:
                options.append((new[0], -1, -1, new[1]))
            if not options:
                continue
            options.sort(key=lambda x: x[0])
            best_delta = options[0][0]
            regret = 0.0
            for ell in range(1, min(k, len(options))):
                regret += options[ell][0] - best_delta
            # tiebreak: larger best_delta (more "urgent")
            score = (regret, options[0][0])
            if (best_overall == (-float("inf"),) or
                score[0] > best_overall[0] or
                (abs(score[0] - best_overall[0]) < 1e-9 and score[1] > best_overall[1])):
                best_overall = score
                chosen = (c, options[0])
        if chosen is None:
            # no insertion possible for any customer -> place each remaining anywhere capacity-feasible
            for c in bank[:]:
                placed = False
                for r in sol.routes:
                    if route_load(r, inst) + inst.customers[c].demand <= inst.fleet.Q(r.vehicle_type) + 1e-6:
                        r.customers.append(c)
                        placed = True
                        break
                if placed:
                    bank.remove(c)
            break
        c, (delta, ri, pos, new_type) = chosen
        if ri == -1:
            sol.routes.append(Route(vehicle_type=new_type, customers=[c]))  # type: ignore[arg-type]
        else:
            sol.routes[ri].customers.insert(pos, c)
        bank.remove(c)
    sol.invalidate_cache()


def vehicle_type_aware_insertion(sol: Solution, removed: List[int], inst: Instance,
                                  rng: random.Random,
                                  tau_frac: float = 0.6,
                                  beta: float = 0.5) -> None:
    """Greedy-style insertion that gives a small bonus when the inserted
    customer pushes a Large vehicle's load above tau_frac * Q_L.
    """
    Q_L = inst.fleet.Q_L
    tau = tau_frac * Q_L

    bank = removed[:]
    while bank:
        best = (float("inf"), -1, -1, -1, None)  # (score, cust, ri, pos, new_type)
        for c in bank:
            cdem = inst.customers[c].demand
            for ri, r in enumerate(sol.routes):
                d, pos = best_insertion(r, c, inst)
                if pos == -1 or d == float("inf"):
                    continue
                bonus = 0.0
                if r.vehicle_type == "L" and route_load(r, inst) + cdem >= tau:
                    bonus = beta
                score = d - bonus
                if score < best[0]:
                    best = (score, c, ri, pos, None)
            new = _try_open_new_route(sol, c, inst)
            if new is not None:
                # bonus if new route is Large and customer's demand alone hits tau (rare with small demand)
                bonus = beta if (new[1] == "L" and cdem >= tau) else 0.0
                score = new[0] - bonus
                if score < best[0]:
                    best = (score, c, -1, -1, new[1])
        score, c, ri, pos, new_type = best
        if c == -1:
            # fall back: place each remaining anywhere capacity-feasible
            for cc in bank[:]:
                for r in sol.routes:
                    if route_load(r, inst) + inst.customers[cc].demand <= inst.fleet.Q(r.vehicle_type) + 1e-6:
                        r.customers.append(cc)
                        bank.remove(cc)
                        break
            break
        if ri == -1:
            sol.routes.append(Route(vehicle_type=new_type, customers=[c]))  # type: ignore[arg-type]
        else:
            sol.routes[ri].customers.insert(pos, c)
        bank.remove(c)
    sol.invalidate_cache()


def regret_2_insertion(sol, removed, inst, rng):
    return regret_k_insertion(sol, removed, inst, rng, k=2)

def regret_3_insertion(sol, removed, inst, rng):
    return regret_k_insertion(sol, removed, inst, rng, k=3)


REPAIR_OPERATORS = [
    ("greedy", greedy_insertion),
    ("regret-2", regret_2_insertion),
    ("regret-3", regret_3_insertion),
    ("type_aware", vehicle_type_aware_insertion),
]
