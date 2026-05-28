"""
Constructive heuristics for the HFVRPTW with soft time windows.

Three options are implemented:

1. construct_type_aware
       Vehicle-type-aware sequential insertion.
       Prefer Large vehicles first, then Small. Within a route, greedily
       insert the cheapest feasible customer.

2. construct_clarke_wright
       Clarke and Wright (1964) parallel savings algorithm, adapted to a
       heterogeneous fleet with two vehicle types. The standard merge
       phase uses the larger capacity Q_L; vehicle types are assigned
       afterwards (large vehicles for the heaviest routes, small for the
       rest), respecting fleet-size limits.

3. construct_random
       Customers are visited in a random order; each customer is appended
       to the first route with enough remaining capacity, otherwise a new
       route is opened. The vehicle type for a new route is chosen at
       random among the types that still have available units.

All three return a fully evaluated Solution.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from .instance import Instance
from .solution import (Route, Solution, best_insertion, route_load,
                       route_distance, route_schedule)


def construct_initial(inst: Instance) -> Solution:
    """Backwards-compatible alias for the vehicle-type-aware insertion."""
    return construct_type_aware(inst)


def construct_type_aware(inst: Instance) -> Solution:
    sol = Solution(routes=[])
    unrouted = list(range(1, len(inst.customers)))  # 1..n
    n_used = {"L": 0, "S": 0}

    while unrouted:
        # choose vehicle type: prefer Large until capacity exhausted
        if n_used["L"] < inst.fleet.m_L:
            v = "L"
        elif n_used["S"] < inst.fleet.m_S:
            v = "S"
        else:
            # No more vehicles available -> pack remaining into the last route as overflow.
            # This shouldn't happen for our parameter settings.
            break

        route = Route(vehicle_type=v, customers=[])
        # greedy fill
        progressed = True
        while progressed and unrouted:
            progressed = False
            best = (float("inf"), -1, -1)  # (delta, customer, position)
            for c in unrouted:
                d, pos = best_insertion(route, c, inst)
                if d < best[0]:
                    best = (d, c, pos)
            delta, c, pos = best
            if c == -1 or delta == float("inf"):
                break
            route.customers.insert(pos, c)
            unrouted.remove(c)
            progressed = True

        if route.customers:
            sol.routes.append(route)
            n_used[v] += 1
        else:
            # Could not insert anything in this vehicle (shouldn't happen for
            # reasonable instances since soft TW only penalizes lateness).
            # Force-insert the first unrouted customer if capacity allows.
            if unrouted:
                c = unrouted[0]
                if inst.customers[c].demand <= inst.fleet.Q(v) + 1e-6:
                    route.customers.append(c)
                    unrouted.remove(c)
                    sol.routes.append(route)
                    n_used[v] += 1
                else:
                    # demand exceeds even Large -> skip vehicle (infeasible problem)
                    break

    sol.evaluate(inst)
    return sol


# ---------------------------------------------------------------------------
# Clarke and Wright savings algorithm
# ---------------------------------------------------------------------------

def _assign_types(routes: List[Route], inst: Instance) -> List[Route]:
    """Assign 'L' or 'S' to each route to respect fleet limits and capacities.

    Strategy: sort routes by total demand (descending). Routes whose load
    exceeds Q_S MUST use 'L'. Among the remaining routes, prefer 'L' for the
    heaviest ones until m_L is exhausted, then assign 'S'. If a route's load
    exceeds Q_L it is left as 'L' (capacity violation will be flagged via
    is_capacity_feasible).
    """
    Q_L = inst.fleet.Q_L
    Q_S = inst.fleet.Q_S
    mL = inst.fleet.m_L
    mS = inst.fleet.m_S
    # sort heavy first
    routes_sorted = sorted(routes, key=lambda r: -route_load(r, inst))
    n_used = {"L": 0, "S": 0}
    for r in routes_sorted:
        load = route_load(r, inst)
        if load > Q_S + 1e-6:
            r.vehicle_type = "L"
            n_used["L"] += 1
        else:
            if n_used["L"] < mL and load > 0.5 * Q_L:
                # heavier among small-feasible routes go to L if room
                r.vehicle_type = "L"
                n_used["L"] += 1
            elif n_used["S"] < mS:
                r.vehicle_type = "S"
                n_used["S"] += 1
            elif n_used["L"] < mL:
                r.vehicle_type = "L"
                n_used["L"] += 1
            else:
                # both fleets exhausted: leave as 'S' (will violate fleet limit)
                r.vehicle_type = "S"
                n_used["S"] += 1
    return routes


def construct_clarke_wright(inst: Instance) -> Solution:
    """Clarke and Wright (1964) parallel savings, adapted to HFVRPTW.

    1. Start with each customer in a singleton route.
    2. Compute savings s(i,j) = d(0,i) + d(0,j) - d(i,j).
    3. Sort savings in non-increasing order.
    4. For each pair (i,j) in order, merge the routes containing i and j
       provided the merged load does not exceed Q_L (the larger capacity)
       and i and j are at the endpoints of their respective routes.
    5. Assign vehicle types ex post (heavy routes -> L, light -> S),
       respecting fleet-size limits.
    """
    n = inst.n
    if n == 0:
        return Solution(routes=[])

    # initial singleton routes (all assumed Large for the merge phase)
    routes: List[List[int]] = [[i] for i in range(1, n + 1)]
    cust2route = {i: idx for idx, i in enumerate(range(1, n + 1))}

    # compute savings
    D = inst.distance
    Q_L = inst.fleet.Q_L
    savings: List[Tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = D[0][i] + D[0][j] - D[i][j]
            savings.append((s, i, j))
    savings.sort(key=lambda t: -t[0])  # descending

    def route_total_demand(route: List[int]) -> float:
        return sum(inst.customers[c].demand for c in route)

    for s, i, j in savings:
        if s <= 0:
            break
        ri = cust2route[i]
        rj = cust2route[j]
        if ri == rj:
            continue
        Ri = routes[ri]
        Rj = routes[rj]
        if not Ri or not Rj:
            continue
        # i must be an endpoint of Ri and j must be an endpoint of Rj
        if i != Ri[0] and i != Ri[-1]:
            continue
        if j != Rj[0] and j != Rj[-1]:
            continue
        # check capacity (largest available type)
        if route_total_demand(Ri) + route_total_demand(Rj) > Q_L + 1e-6:
            continue
        # orient and merge: end of Ri --- start of Rj
        if i == Ri[0]:
            Ri.reverse()
        if j == Rj[-1]:
            Rj.reverse()
        # now i is at the end of Ri, j is at the start of Rj
        merged = Ri + Rj
        routes[ri] = merged
        routes[rj] = []
        for c in Rj:
            cust2route[c] = ri

    # build Route objects (initially mark all as 'L', will reassign types next)
    raw_routes = [Route(vehicle_type="L", customers=r[:]) for r in routes if r]
    # type assignment: large for heavy routes, small for the rest
    _assign_types(raw_routes, inst)

    sol = Solution(routes=raw_routes)
    sol.evaluate(inst)
    return sol


# ---------------------------------------------------------------------------
# TW-aware Clarke and Wright (simple): Solomon-style temporal compatibility
# ---------------------------------------------------------------------------

def construct_clarke_wright_tw_simple(inst: Instance, mu: float = 1.0) -> Solution:
    """TW-aware CW with a simple temporal-compatibility penalty in savings.

    The saving for merging customer i (end of one route) and j (start of
    another) is

        s'(i, j) = s(i, j) - mu * temporal_gap(i, j)

    where temporal_gap(i, j) measures how incompatible the windows are
    along the direction i -> j:

        temporal_gap(i, j) = max(0, a_j - (b_i + s_i + t_{ij}))   [too early -> wait, OK]
                           + max(0, (a_i + s_i + t_{ij}) - b_j)   [too late at j]

    Only the second term truly hurts (waiting is free in cost terms but a
    huge gap before the next visit usually means a low-quality route).
    Pairs with a positive penalized saving are merged in non-increasing
    order of s'(i,j). Otherwise this is identical to the standard CW.
    """
    n = inst.n
    if n == 0:
        return Solution(routes=[])

    routes: List[List[int]] = [[i] for i in range(1, n + 1)]
    cust2route = {i: idx for idx, i in enumerate(range(1, n + 1))}

    D = inst.distance
    T = inst.travel_time
    Q_L = inst.fleet.Q_L

    def temporal_gap(i: int, j: int) -> float:
        ci = inst.customers[i]
        cj = inst.customers[j]
        # earliest possible service start at i, then go to j
        depart_i = max(ci.ready, D[0][i]) + ci.service
        arrival_j = depart_i + T[i][j]
        # waiting at j (free) vs being late at j (bad)
        late_at_j = max(0.0, arrival_j - cj.due)
        # symmetric "wait gap" (penalize huge waiting)
        wait_at_j = max(0.0, cj.ready - arrival_j)
        return late_at_j + 0.25 * wait_at_j

    savings: List[Tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            s = D[0][i] + D[0][j] - D[i][j]
            s_prime = s - mu * temporal_gap(i, j)
            savings.append((s_prime, i, j))
    savings.sort(key=lambda t: -t[0])

    def total_demand(route: List[int]) -> float:
        return sum(inst.customers[c].demand for c in route)

    for s_prime, i, j in savings:
        if s_prime <= 0:
            break
        ri = cust2route[i]
        rj = cust2route[j]
        if ri == rj:
            continue
        Ri = routes[ri]
        Rj = routes[rj]
        if not Ri or not Rj:
            continue
        # require: i is the LAST customer of Ri, j is the FIRST of Rj (ordered savings)
        if Ri[-1] != i or Rj[0] != j:
            continue
        if total_demand(Ri) + total_demand(Rj) > Q_L + 1e-6:
            continue
        merged = Ri + Rj
        routes[ri] = merged
        routes[rj] = []
        for c in Rj:
            cust2route[c] = ri

    raw_routes = [Route(vehicle_type="L", customers=r[:]) for r in routes if r]
    _assign_types(raw_routes, inst)
    sol = Solution(routes=raw_routes)
    sol.evaluate(inst)
    return sol


# ---------------------------------------------------------------------------
# TW-aware Clarke and Wright (full): real cost-delta with soft-TW lateness
# ---------------------------------------------------------------------------

def _route_partial_cost(customers: List[int], inst: Instance,
                        vehicle_type: str = "L") -> float:
    """Distance cost (alpha_v * dist) + lateness penalty for a tentative route."""
    if not customers:
        return 0.0
    r = Route(vehicle_type=vehicle_type, customers=customers)
    d = route_distance(r, inst)
    _, late = route_schedule(r, inst)
    return inst.fleet.alpha(vehicle_type) * d + inst.lateness_penalty * late


def construct_clarke_wright_tw_full(inst: Instance) -> Solution:
    """TW-aware CW that uses the *real* cost-delta as the savings measure.

    At every iteration we look at all pairs of routes (R_p, R_q) and all
    four orientations (head/tail of one against head/tail of the other),
    compute the actual reduction in (distance cost + lateness penalty)
    that the merge would yield, and pick the best (largest reduction).
    Iteration stops when no merge yields a positive reduction or the
    capacity constraint blocks all merges.

    This is more expensive than vanilla CW (O(R^2) per iteration with
    R routes still alive) but for small instances (n <= 50) it is
    negligible and produces a much better initial solution under soft TW.
    """
    n = inst.n
    if n == 0:
        return Solution(routes=[])

    routes: List[List[int]] = [[i] for i in range(1, n + 1)]
    Q_L = inst.fleet.Q_L

    def total_demand(route: List[int]) -> float:
        return sum(inst.customers[c].demand for c in route)

    # cache cost of each route
    cost_cache: List[float] = [_route_partial_cost(r, inst, "L") for r in routes]

    while True:
        best = (0.0, -1, -1, None)  # (reduction, pi, qi, merged_seq)
        for p in range(len(routes)):
            Rp = routes[p]
            if not Rp:
                continue
            for q in range(len(routes)):
                if p == q:
                    continue
                Rq = routes[q]
                if not Rq:
                    continue
                if total_demand(Rp) + total_demand(Rq) > Q_L + 1e-6:
                    continue
                # try four merges: Rp+Rq, Rp+rev(Rq), rev(Rp)+Rq, rev(Rp)+rev(Rq)
                # rev() only changes route orientation if length > 1
                candidates = [Rp + Rq]
                if len(Rq) > 1:
                    candidates.append(Rp + Rq[::-1])
                if len(Rp) > 1:
                    candidates.append(Rp[::-1] + Rq)
                if len(Rp) > 1 and len(Rq) > 1:
                    candidates.append(Rp[::-1] + Rq[::-1])
                base_cost = cost_cache[p] + cost_cache[q]
                # also pay 1 fewer fixed vehicle cost when merging two routes
                base_cost_with_fixed = base_cost + inst.fleet.F_L  # approximation
                for seq in candidates:
                    new_cost = _route_partial_cost(seq, inst, "L")
                    new_cost_with_fixed = new_cost  # only one route now: 1x F_L
                    reduction = base_cost_with_fixed - (new_cost + inst.fleet.F_L) \
                                + inst.fleet.F_L  # save one F_L
                    # Simplify: reduction = (Cp + Cq + 2 F_L) - (Cmerged + 1 F_L)
                    #                   = Cp + Cq + F_L - Cmerged
                    reduction = (cost_cache[p] + cost_cache[q]
                                 + inst.fleet.F_L
                                 - new_cost)
                    if reduction > best[0] + 1e-9:
                        best = (reduction, p, q, seq)
        if best[1] == -1:
            break
        _, p, q, merged = best
        routes[p] = merged
        routes[q] = []
        cost_cache[p] = _route_partial_cost(merged, inst, "L")
        cost_cache[q] = 0.0

    raw_routes = [Route(vehicle_type="L", customers=r[:]) for r in routes if r]
    _assign_types(raw_routes, inst)
    sol = Solution(routes=raw_routes)
    sol.evaluate(inst)
    return sol


# ---------------------------------------------------------------------------
# Random initial solution
# ---------------------------------------------------------------------------

def construct_random(inst: Instance, seed: int = 0) -> Solution:
    """Random initial solution.

    Customers are visited in a random order; each is appended to the first
    existing route with enough remaining capacity (its assigned vehicle
    type's Q_v), otherwise a new route is opened with a randomly chosen
    vehicle type that still has available units. Final type assignment is
    refined with _assign_types to respect fleet-size limits.
    """
    rng = random.Random(seed)
    n_used = {"L": 0, "S": 0}
    order = list(range(1, len(inst.customers)))
    rng.shuffle(order)
    routes: List[Route] = []

    for c in order:
        cdem = inst.customers[c].demand
        # try existing routes (random scan order)
        idxs = list(range(len(routes)))
        rng.shuffle(idxs)
        placed = False
        for ri in idxs:
            r = routes[ri]
            cap = inst.fleet.Q(r.vehicle_type)
            if route_load(r, inst) + cdem <= cap + 1e-6:
                # insert at random position
                pos = rng.randint(0, len(r.customers))
                r.customers.insert(pos, c)
                placed = True
                break
        if placed:
            continue
        # open a new route with a randomly chosen available type
        avail = []
        if n_used["L"] < inst.fleet.m_L and cdem <= inst.fleet.Q_L + 1e-6:
            avail.append("L")
        if n_used["S"] < inst.fleet.m_S and cdem <= inst.fleet.Q_S + 1e-6:
            avail.append("S")
        if not avail:
            # force into the most permissive existing type with capacity even if violating
            # (shouldn't happen for typical instances)
            v = "L" if cdem <= inst.fleet.Q_L + 1e-6 else "S"
        else:
            v = rng.choice(avail)
        routes.append(Route(vehicle_type=v, customers=[c]))
        n_used[v] += 1

    # post-hoc type rebalancing (heavy routes -> L)
    _assign_types(routes, inst)

    sol = Solution(routes=routes)
    sol.evaluate(inst)
    return sol
