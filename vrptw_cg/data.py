"""Instance loading and preprocessing for the Solomon VRPTW benchmark.

Node convention (kept identical to the original implementation so that the
`solomon-instances/` and `routes/` data files stay compatible):

    node 0       -> depot (route start copy)
    node 1..n    -> customers
    node n + 1   -> depot (route end copy, same coordinates/demand as node 0)

All distances are Euclidean, rounded to the nearest integer, which is the
convention used by exact-method papers on the Solomon benchmark (see
Desrochers, Desrosiers & Solomon, 1992) so that travel time and travel cost
can share the same numeric value.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

FIELDS = ("CUST-NO.", "XCOORD.", "YCOORD.", "DEMAND", "READY-TIME",
          "DUE-DATE", "SERVICE-TIME")

SOLOMON_DIR = "solomon-instances"


@dataclass
class Instance:
    """A VRPTW instance with the depot duplicated as node 0 and node n+1."""

    name: str
    n: int                      # number of customers (excludes both depot copies)
    K: int                      # number of available vehicles (fleet size)
    Q: float                    # vehicle capacity
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray                # demand, length n+2
    a: np.ndarray                # ready time (post time-window reduction if requested)
    b: np.ndarray                # due date  (post time-window reduction if requested)
    a_raw: np.ndarray            # original ready time, before reduction
    b_raw: np.ndarray            # original due date, before reduction
    d: np.ndarray                # (n+2) x (n+2) integer distance/time matrix
    neighbors: dict = field(default_factory=dict)  # ng-route neighbor sets

    @property
    def num_nodes(self) -> int:
        return self.n + 2

    def build_neighbors(self, k: int = 8) -> None:
        """Compute the k-nearest-customer neighbor sets used by ng-route
        relaxation pricing (Baldacci, Mingozzi & Roberti, 2011).

        neighbors[j] is a frozenset of the k customers nearest to customer j
        (j itself excluded). The depot copies do not need a neighbor set.
        """
        neighbors = {}
        for j in range(1, self.n + 1):
            order = sorted((c for c in range(1, self.n + 1) if c != j),
                            key=lambda c: self.d[j, c])
            neighbors[j] = frozenset(order[:k])
        self.neighbors = neighbors


def _read_raw_rows(filename: str):
    with open(filename, "r") as fh:
        stream = fh.readlines()
    if not stream:
        raise ValueError(f"Empty or unreadable instance file: {filename}")

    vehicle_number, capacity = (int(v) for v in stream[4].split())

    rows = []
    for line in stream[9:]:
        if line.strip() == "":
            continue
        values = line.split()
        if len(values) != len(FIELDS):
            continue
        rows.append(dict(zip(FIELDS, values)))
    return vehicle_number, capacity, rows


def load_instance(instance_name: str, n_customers: int,
                   solomon_dir: str = SOLOMON_DIR,
                   reduce_time_windows: bool = True,
                   neighbor_k: int = 8) -> Instance:
    """Load a Solomon instance, truncated to its first ``n_customers`` rows.

    This replaces the interactive ``input()``-driven loader from the
    original codebase so instances can be selected from the CLI or a
    benchmark script instead of typed in by hand.
    """
    if instance_name.endswith(".txt"):
        instance_name = instance_name[:-4]
    filename = os.path.join(solomon_dir, instance_name + ".txt")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Instance not found: {filename}")
    if not (1 <= n_customers <= 100):
        raise ValueError("n_customers must be between 1 and 100")

    K, Q, rows = _read_raw_rows(filename)
    rows = rows[: n_customers + 1]          # depot + n customers
    depot_copy = dict(rows[0])
    depot_copy["CUST-NO."] = str(n_customers + 1)
    rows.append(depot_copy)

    x = np.array([int(r["XCOORD."]) for r in rows], dtype=float)
    y = np.array([int(r["YCOORD."]) for r in rows], dtype=float)
    q = np.array([int(r["DEMAND"]) for r in rows], dtype=float)
    a_raw = np.array([int(r["READY-TIME"]) for r in rows], dtype=float)
    b_raw = np.array([int(r["DUE-DATE"]) for r in rows], dtype=float)

    d = _distance_matrix(x, y)

    if reduce_time_windows:
        a, b = _reduce_time_windows(n_customers, d, a_raw, b_raw)
    else:
        a, b = a_raw.copy(), b_raw.copy()

    inst = Instance(name=instance_name, n=n_customers, K=K, Q=Q, x=x, y=y,
                     q=q, a=a, b=b, a_raw=a_raw, b_raw=b_raw, d=d)
    inst.build_neighbors(k=neighbor_k)
    return inst


def _distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(x[i] - x[j], y[i] - y[j])
            d[i, j] = d[j, i] = int(round(dist))
    return d


def _reduce_time_windows(n: int, d: np.ndarray, ready: np.ndarray,
                          due: np.ndarray):
    """Desrochers-style time-window tightening (Desrochers, Desrosiers &
    Solomon, 1992, Prop. 3). Kept from the original implementation: it is a
    correct and standard exact-method preprocessing step, not a heuristic.
    """
    a = ready.copy()
    b = due.copy()
    updated = True
    while updated:
        updated = False
        for k in range(1, n + 1):
            min_arr_pred = min([b[k],
                                 min(a[i] + d[i, k] for i in range(n + 1) if i != k)])
            min_arr_next = min([b[k],
                                 min(a[j] - d[k, j] for j in range(1, n + 2) if j != k)])
            new_a = max([a[k], min_arr_pred, min_arr_next])
            if new_a != a[k]:
                updated = True
            a[k] = new_a

            max_dep_pred = max([a[k],
                                 max(b[i] + d[i, k] for i in range(n + 1) if i != k)])
            max_dep_next = max([a[k],
                                 max(b[j] - d[k, j] for j in range(1, n + 2) if j != k)])
            new_b = min([b[k], max_dep_pred, max_dep_next])
            if new_b != b[k]:
                updated = True
            b[k] = new_b
    return a, b


def route_cost(route, d: np.ndarray) -> float:
    return sum(d[route[i], route[i + 1]] for i in range(len(route) - 1))


def route_load(route, q: np.ndarray) -> float:
    return sum(q[node] for node in route[1:-1])
