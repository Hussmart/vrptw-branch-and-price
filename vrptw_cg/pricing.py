"""ng-route relaxation pricing (the column-generation subproblem).

The subproblem is an Elementary Shortest Path Problem with Resource
Constraints (ESPPRC): find negative-reduced-cost paths from the depot (node
0) to the depot copy (node n+1) respecting vehicle capacity and time
windows, and *without repeating a customer*.

The original implementation solved the *non*-elementary relaxation (a plain
resource-indexed shortest path, Desrochers et al. 1992) which is fast but
allows a route to revisit a customer, and a full elementary MIP (`ESPmodel`
in the legacy/ folder) which is provably exact but times out in minutes even
on small instances (see legacy/note.txt).

ng-route relaxation (Baldacci, Mingozzi & Roberti, 2011; refined by
Bulhoes, Sadykov & Uchoa, 2018 and used inside the bidirectional bucket-graph
labeling algorithm of Sadykov, Uchoa & Pessoa, 2021 -- the state of the art)
sits between the two: each customer j only remembers a small "memory" set
bounded by its own k-nearest-neighbor list, so short/likely cycles are
forbidden while the label state space stays polynomially small. It is the
standard building block of every modern VRP branch-cut-and-price code
(BaPCod/VRPSolver, RouteOpt, ...).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NEG_TOL = -1e-6


@dataclass(frozen=True)
class Label:
    node: int
    cost: float
    load: float
    time: float
    memory: frozenset
    path: tuple


def _dominates(l1: Label, l2: Label) -> bool:
    """True if l1 makes l2 useless: equal-or-better on every resource and a
    subset (i.e. weaker/easier-to-extend) memory set."""
    if l1 is l2:
        return False
    return (l1.cost <= l2.cost + 1e-9 and l1.load <= l2.load + 1e-9 and
            l1.time <= l2.time + 1e-9 and l1.memory <= l2.memory)


class PricingResult:
    def __init__(self, routes: list[tuple], reduced_costs: list[float],
                 labels_processed: int, truncated: bool):
        self.routes = routes
        self.reduced_costs = reduced_costs
        self.labels_processed = labels_processed
        self.truncated = truncated

    def __bool__(self):
        return len(self.routes) > 0


def solve_pricing(instance, duals_customers: np.ndarray, dual_fleet: float,
                   arc_forbidden: Optional[set] = None,
                   arc_required_next: Optional[dict] = None,
                   arc_required_prev: Optional[dict] = None,
                   pair_together: Optional[frozenset] = None,
                   pair_apart: Optional[frozenset] = None,
                   max_routes: int = 25,
                   max_labels: int = 200_000) -> PricingResult:
    """Label-setting ng-route pricing.

    Returns up to ``max_routes`` distinct negative-reduced-cost routes,
    sorted by reduced cost (most negative first), which is the standard
    "multiple columns per iteration" speed-up for column generation.

    ``arc_forbidden`` / ``arc_required_next`` / ``arc_required_prev`` encode
    the arc-branching decisions of the enclosing branch-and-price node (see
    branch_and_price.py): forbidding or forcing a specific arc (i, j).

    ``pair_together`` / ``pair_apart`` encode Ryan-Foster branching decisions
    (Ryan & Foster, 1981): pairs of customers that must appear in the same
    route, or must never appear in the same route. These are cheapest to
    enforce as a post-filter on *completed* paths (checked once a route
    reaches the sink) rather than inside the label extension itself, since
    "customer q is visited later in this same route" cannot be decided from
    an intermediate label without extra bookkeeping; correctness is
    unaffected, only some dominance pruning opportunities are missed.
    """
    n = instance.n
    d = instance.d
    a, b = instance.a, instance.b
    q = instance.q
    Q = instance.Q
    N = instance.neighbors
    sink = n + 1

    arc_forbidden = arc_forbidden or set()
    arc_required_next = arc_required_next or {}
    arc_required_prev = arc_required_prev or {}

    # pi[0] and pi[sink] are 0 (the depot rows do not have a covering
    # constraint); dual_fleet is charged once per route, at departure.
    pi = np.zeros(n + 2)
    pi[1:n + 1] = duals_customers

    start = Label(node=0, cost=-dual_fleet, load=0.0, time=float(a[0]),
                  memory=frozenset(), path=(0,))
    labels_at: dict[int, list[Label]] = {i: [] for i in range(n + 2)}
    labels_at[0].append(start)
    frontier = deque([start])
    processed = 0
    truncated = False

    def allowed(i: int, j: int) -> bool:
        if (i, j) in arc_forbidden:
            return False
        req_j = arc_required_next.get(i)
        if req_j is not None and req_j != j:
            return False
        req_i = arc_required_prev.get(j)
        if req_i is not None and req_i != i:
            return False
        return True

    while frontier:
        label = frontier.popleft()
        processed += 1
        if processed > max_labels:
            truncated = True
            break
        i = label.node
        if i == sink:
            continue

        for j in range(1, n + 2):
            if j == i or not allowed(i, j):
                continue
            if j <= n and j in label.memory:
                continue  # ng-route elementarity rule

            new_load = label.load + (q[j] if j <= n else 0.0)
            if new_load > Q + 1e-9:
                continue
            arrival = label.time + d[i, j]
            new_time = max(arrival, a[j])
            if new_time > b[j] + 1e-9:
                continue

            new_cost = label.cost + (d[i, j] - pi[i])
            new_memory = (label.memory & N.get(j, frozenset())) | {j} if j <= n \
                else frozenset()
            new_label = Label(j, new_cost, new_load, new_time, new_memory,
                               label.path + (j,))

            bucket = labels_at[j]
            if any(_dominates(existing, new_label) for existing in bucket):
                continue
            labels_at[j] = [l for l in bucket if not _dominates(new_label, l)]
            labels_at[j].append(new_label)
            frontier.append(new_label)

    finished = [l for l in labels_at[sink] if l.cost < NEG_TOL]
    finished.sort(key=lambda l: l.cost)

    def respects_pairs(path: tuple) -> bool:
        visited = set(path)
        if pair_together:
            for p, qc in pair_together:
                if (p in visited) != (qc in visited):
                    return False
        if pair_apart:
            for p, qc in pair_apart:
                if p in visited and qc in visited:
                    return False
        return True

    routes, costs, seen = [], [], set()
    for label in finished:
        if label.path in seen or not respects_pairs(label.path):
            continue
        seen.add(label.path)
        routes.append(label.path)
        costs.append(label.cost)
        if len(routes) >= max_routes:
            break

    return PricingResult(routes, costs, processed, truncated)
