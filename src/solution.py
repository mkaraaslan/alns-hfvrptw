"""
Solution representation, cost evaluation, and feasibility checks for the
HFVRPTW with soft time windows.

A solution is a list of Route objects. Each Route has:
- a vehicle type ('L' or 'S')
- an ordered list of customer indices (excluding the depot at start/end)

Constraints:
- Capacity: HARD (load <= Q_v).
- Time windows: SOFT for late arrivals -- being late is allowed but penalized.
                Early arrival waits until ready time (standard).
- Fleet size limits: HARD (count of routes per type <= m_v).

Cost = sum_routes [F_v + alpha_v * dist(R)] + lateness_penalty * sum_late_minutes
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Tuple

from .instance import Instance


@dataclass
class Route:
    vehicle_type: str            # 'L' or 'S'
    customers: List[int] = field(default_factory=list)  # customer indices (no depot)

    def __len__(self) -> int:
        return len(self.customers)

    def copy(self) -> "Route":
        return Route(vehicle_type=self.vehicle_type, customers=self.customers[:])


# --------------------------------------------------------------------------
# Per-route metrics: distance, load, total lateness (soft TW)
# --------------------------------------------------------------------------

def route_load(route: Route, inst: Instance) -> float:
    return sum(inst.customers[i].demand for i in route.customers)


def route_distance(route: Route, inst: Instance) -> float:
    if not route.customers:
        return 0.0
    D = inst.distance
    seq = [0] + route.customers + [0]
    total = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        total += D[a][b]
    return total


def route_schedule(route: Route, inst: Instance) -> Tuple[List[float], float]:
    """Compute service start times along the route (waiting at early arrival).

    Returns (start_times, total_lateness):
        start_times[k] = service start time at the k-th node of the route
                          (including depot start = 0 and depot return).
        total_lateness = sum over customers of max(0, start_time - due_i).
    """
    if not route.customers:
        return [0.0, 0.0], 0.0
    T = inst.travel_time
    seq = [0] + route.customers + [0]
    starts: List[float] = [0.0] * len(seq)  # depot leaves at time 0
    lateness = 0.0
    cur_time = 0.0
    for k in range(1, len(seq)):
        prev = seq[k - 1]
        cust_idx = seq[k]
        cust = inst.customers[cust_idx]
        # travel
        cur_time = cur_time + T[prev][cust_idx]
        # finish previous service before traveling: when prev != depot, add service time
        # We handle service time on departure side: starts[k] = arrival_time
        # but we need to add prev service time as well for k>=2:
        if k >= 2:
            prev_cust = inst.customers[prev]
            cur_time = cur_time + prev_cust.service
        # wait if early
        if cust_idx != 0:
            if cur_time < cust.ready:
                cur_time = cust.ready
            starts[k] = cur_time
            # lateness: only for customers (not depot return)
            if cur_time > cust.due:
                lateness += cur_time - cust.due
        else:
            # arrival at depot (return); we just record arrival time (no waiting needed for cost,
            # but Solomon depot has its own due time which we treat as soft as well).
            starts[k] = cur_time
            depot = inst.customers[0]
            if cur_time > depot.due:
                lateness += cur_time - depot.due

    return starts, lateness


def is_route_capacity_feasible(route: Route, inst: Instance) -> bool:
    return route_load(route, inst) <= inst.fleet.Q(route.vehicle_type) + 1e-6


def route_cost(route: Route, inst: Instance) -> Tuple[float, float, float]:
    """Return (vehicle+distance cost, lateness, total_route_cost_with_penalty).

    The vehicle+distance cost is F_v + alpha_v * dist(R) (only if the route is non-empty).
    """
    if not route.customers:
        return 0.0, 0.0, 0.0
    v = route.vehicle_type
    fleet = inst.fleet
    dist = route_distance(route, inst)
    fixed_var = fleet.F(v) + fleet.alpha(v) * dist
    _, lateness = route_schedule(route, inst)
    pen = inst.lateness_penalty * lateness
    return fixed_var, lateness, fixed_var + pen


# --------------------------------------------------------------------------
# Solution
# --------------------------------------------------------------------------

@dataclass
class Solution:
    routes: List[Route] = field(default_factory=list)

    # Cached cost components (recomputed lazily)
    _cost: float | None = field(default=None, repr=False, compare=False)
    _vehicle_cost: float | None = field(default=None, repr=False, compare=False)
    _distance_cost: float | None = field(default=None, repr=False, compare=False)
    _lateness: float | None = field(default=None, repr=False, compare=False)

    def copy(self) -> "Solution":
        s = Solution(routes=[r.copy() for r in self.routes])
        return s

    def invalidate_cache(self) -> None:
        self._cost = None
        self._vehicle_cost = None
        self._distance_cost = None
        self._lateness = None

    def evaluate(self, inst: Instance) -> float:
        total = 0.0
        veh = 0.0
        dist_cost = 0.0
        late = 0.0
        for r in self.routes:
            if not r.customers:
                continue
            v = r.vehicle_type
            d = route_distance(r, inst)
            veh += inst.fleet.F(v)
            dist_cost += inst.fleet.alpha(v) * d
            _, l = route_schedule(r, inst)
            late += l
        total = veh + dist_cost + inst.lateness_penalty * late
        self._cost = total
        self._vehicle_cost = veh
        self._distance_cost = dist_cost
        self._lateness = late
        return total

    @property
    def cost(self) -> float:
        assert self._cost is not None, "call evaluate(inst) first"
        return self._cost

    def visited_customers(self) -> List[int]:
        out: List[int] = []
        for r in self.routes:
            out.extend(r.customers)
        return out

    def routes_of_type(self, v: str) -> List[Route]:
        return [r for r in self.routes if r.vehicle_type == v and r.customers]

    def count_used(self, v: str) -> int:
        return len(self.routes_of_type(v))

    def is_capacity_feasible(self, inst: Instance) -> bool:
        return all(is_route_capacity_feasible(r, inst) for r in self.routes)

    def is_fleet_feasible(self, inst: Instance) -> bool:
        return (self.count_used("L") <= inst.fleet.m_L
                and self.count_used("S") <= inst.fleet.m_S)

    def total_lateness(self, inst: Instance) -> float:
        if self._lateness is None:
            self.evaluate(inst)
        return self._lateness  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Insertion utilities
# --------------------------------------------------------------------------

def insertion_delta(route: Route, customer: int, position: int,
                    inst: Instance) -> Tuple[float, bool]:
    """Compute the cost delta of inserting customer at the given position
    (so it becomes route.customers[position]).

    Returns (delta_cost, capacity_feasible). The delta accounts for vehicle
    fixed cost (route was empty before -> add F_v if becomes non-empty),
    distance change scaled by alpha_v, plus the change in lateness penalty.
    Soft TW: insertion is always "feasible" as long as capacity holds; the
    cost reflects any added lateness.
    """
    fleet = inst.fleet
    v = route.vehicle_type

    # Capacity check (hard)
    new_load = route_load(route, inst) + inst.customers[customer].demand
    if new_load > fleet.Q(v) + 1e-6:
        return float("inf"), False

    # Build a temporary route to evaluate cost
    new_customers = route.customers[:position] + [customer] + route.customers[position:]
    new_route = Route(vehicle_type=v, customers=new_customers)
    new_dist = route_distance(new_route, inst)
    _, new_late = route_schedule(new_route, inst)

    if route.customers:
        old_dist = route_distance(route, inst)
        _, old_late = route_schedule(route, inst)
        delta = (fleet.alpha(v) * (new_dist - old_dist)
                 + inst.lateness_penalty * (new_late - old_late))
    else:
        # opening a new route: add fixed cost + new distance cost + new lateness
        delta = fleet.F(v) + fleet.alpha(v) * new_dist + inst.lateness_penalty * new_late

    return delta, True


def best_insertion(route: Route, customer: int, inst: Instance) -> Tuple[float, int]:
    """Find the best position to insert customer in route.

    Returns (best_delta, best_position). If capacity-infeasible everywhere,
    returns (inf, -1).
    """
    best = (float("inf"), -1)
    for pos in range(len(route.customers) + 1):
        d, ok = insertion_delta(route, customer, pos, inst)
        if ok and d < best[0]:
            best = (d, pos)
    return best
