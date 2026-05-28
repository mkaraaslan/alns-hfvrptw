"""
Solomon-style instance loader for the Heterogeneous Fleet VRPTW (HFVRPTW)
with soft time windows.

The original Solomon files have one vehicle type (capacity Q) per instance.
We load the customer/depot data and overlay a heterogeneous fleet configuration
(two vehicle types: Large = L, Small = S) with different capacities, fixed
costs, and per-distance variable costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Customer:
    idx: int          # 0 = depot
    x: float
    y: float
    demand: float
    ready: float      # a_i
    due: float        # b_i
    service: float    # s_i


@dataclass
class FleetConfig:
    """Two vehicle types: Large (L) and Small (S).

    F_v: fixed cost per used vehicle of type v.
    alpha_v: per-distance variable cost of type v.
    Q_v: capacity of type v.
    m_v: maximum number of vehicles of type v available.
    """
    Q_L: float = 200.0
    F_L: float = 200.0
    alpha_L: float = 1.0
    m_L: int = 3

    Q_S: float = 100.0
    F_S: float = 100.0
    alpha_S: float = 1.5
    m_S: int = 4

    def Q(self, v: str) -> float:
        return self.Q_L if v == "L" else self.Q_S

    def F(self, v: str) -> float:
        return self.F_L if v == "L" else self.F_S

    def alpha(self, v: str) -> float:
        return self.alpha_L if v == "L" else self.alpha_S

    def m(self, v: str) -> int:
        return self.m_L if v == "L" else self.m_S


@dataclass
class Instance:
    name: str
    customers: List[Customer]
    fleet: FleetConfig
    # soft time-window penalty (per unit of late time, summed over customers)
    lateness_penalty: float = 100.0
    distance: List[List[float]] = field(default_factory=list)
    travel_time: List[List[float]] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Number of customers (excluding depot)."""
        return len(self.customers) - 1

    @property
    def depot(self) -> Customer:
        return self.customers[0]

    def __post_init__(self):
        if not self.distance:
            self.distance = self._build_distance_matrix()
        if not self.travel_time:
            # Solomon convention: travel time == Euclidean distance (rounded if needed)
            self.travel_time = [row[:] for row in self.distance]

    def _build_distance_matrix(self) -> List[List[float]]:
        N = len(self.customers)
        D = [[0.0] * N for _ in range(N)]
        for i in range(N):
            ci = self.customers[i]
            for j in range(N):
                if i == j:
                    continue
                cj = self.customers[j]
                D[i][j] = math.hypot(ci.x - cj.x, ci.y - cj.y)
        return D


def load_solomon(path: str | Path, fleet: FleetConfig | None = None,
                 lateness_penalty: float = 100.0) -> Instance:
    """Load a Solomon-style instance file.

    The format is the standard Solomon layout:
        line 1: instance name
        ...
        VEHICLE
        NUMBER  CAPACITY
        <m>     <Q>
        ...
        CUSTOMER
        CUST NO. XCOORD. YCOORD. DEMAND READY TIME DUE DATE SERVICE TIME
        0  ... (depot)
        1  ...
        ...
    """
    path = Path(path)
    raw = path.read_text().splitlines()

    # find the customer block: it is the contiguous block of numeric lines
    # after the header line containing "CUST NO." (or the last line starting
    # with "CUST" or "CUSTOMER")
    customers: List[Customer] = []
    in_customer_block = False
    for line in raw:
        stripped = line.strip()
        if not stripped:
            continue
        # heuristic: a customer line starts with an integer
        tokens = stripped.split()
        if not in_customer_block:
            # look for the column header that has CUST and XCOORD
            upper = stripped.upper()
            if upper.startswith("CUST NO") or ("CUST NO." in upper) or \
               (("CUST" in upper) and ("XCOORD" in upper)):
                in_customer_block = True
            continue
        # we are inside the customer block: parse if numeric
        try:
            idx = int(tokens[0])
        except (ValueError, IndexError):
            continue
        if len(tokens) < 7:
            continue
        x, y, dem, rdy, due, srv = (float(t) for t in tokens[1:7])
        customers.append(Customer(idx=idx, x=x, y=y, demand=dem,
                                   ready=rdy, due=due, service=srv))

    if not customers:
        raise ValueError(f"No customers parsed from {path}")
    customers.sort(key=lambda c: c.idx)

    name = path.stem
    fleet = fleet if fleet is not None else FleetConfig()
    return Instance(name=name, customers=customers, fleet=fleet,
                    lateness_penalty=lateness_penalty)
