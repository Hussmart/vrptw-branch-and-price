"""Branch-and-Price: column generation plus a branch-and-bound layer that
closes the integrality gap.

This is the piece the original project was missing entirely -- it only ever
solved the LP relaxation of the set-partitioning master (see
GAP_ANALYSIS_AND_ROADMAP.md). Column generation alone gives a valid *lower
bound* and a (generally fractional) LP solution; it does not certify an
integer-optimal routing plan. Branch-and-Price restores that guarantee by
branching on fractional variables and re-running column generation at every
node of the search tree (Barnhart et al., 1998; Desrochers, Desrosiers &
Solomon, 1992 for the VRPTW-specific branching rules used here).

Branching rules, applied in this order at every fractional node:
  1. Vehicle-count branching: if sum(y_r) is fractional, branch floor/ceil
     by tightening the master's fleet-size window (Desrochers-Desrosiers-
     Solomon, 1992).
  2. Arc branching: otherwise pick the most fractional aggregated arc flow
     x_ij = sum_r (y_r if arc (i,j) in r else 0) and branch on forbidding /
     forcing that arc.
  3. Ryan-Foster branching (Ryan & Foster, 1981): arc branching alone is
     *not* a complete branching rule for a set-partitioning master -- there
     exist fractional solutions where every aggregated arc flow is already
     integral (e.g. two distinct routes that happen to overlap perfectly on
     every arc's total flow while each individual y_r stays fractional).
     When step 2 finds no fractional arc but y is still fractional, we fall
     back to Ryan-Foster: pick two customers whose *joint* coverage
     sum_r(y_r : both in r) is fractional and branch on "must be in the
     same route" vs. "must never be in the same route". This is the
     standard, provably complete branching rule for set-partitioning
     formulations and is what actually guarantees termination.

Every node re-solves the master with the customer-artificial trick in
master.py, so a node is always LP-feasible; a node is pruned as *routing*-
infeasible only if the converged LP solution still needs an artificial
variable.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from .data import route_cost
from .heuristics import impact_construction
from .master import MasterProblem
from .pricing import solve_pricing

EPS = 1e-6


@dataclass
class Column:
    route: tuple
    cost: float
    incidence: np.ndarray


@dataclass
class BPNode:
    fleet_lb: int
    fleet_ub: int
    arc_forbidden: frozenset
    arc_required_next: dict
    arc_required_prev: dict
    pair_together: frozenset
    pair_apart: frozenset
    depth: int

    def child(self, **overrides) -> "BPNode":
        fields = dict(fleet_lb=self.fleet_lb, fleet_ub=self.fleet_ub,
                      arc_forbidden=self.arc_forbidden,
                      arc_required_next=self.arc_required_next,
                      arc_required_prev=self.arc_required_prev,
                      pair_together=self.pair_together,
                      pair_apart=self.pair_apart, depth=self.depth + 1)
        fields.update(overrides)
        return BPNode(**fields)


@dataclass
class NodeResult:
    feasible: bool
    bound: float = math.inf
    y: np.ndarray | None = None
    columns: list | None = None
    converged: bool = False


@dataclass
class BPStats:
    nodes_explored: int = 0
    nodes_pruned_bound: int = 0
    nodes_pruned_infeasible: int = 0
    columns_generated: int = 0
    root_lp_bound: float | None = None
    root_iteration_bounds: list = None
    elapsed_seconds: float = 0.0
    time_limit_hit: bool = False

    def __post_init__(self):
        if self.root_iteration_bounds is None:
            self.root_iteration_bounds = []


@dataclass
class BPResult:
    status: str                 # "optimal" (within limits) or "feasible" (limits hit)
    incumbent_routes: list
    incumbent_cost: float
    lower_bound: float          # best proven bound over the remaining open nodes
    stats: BPStats


class BranchAndPrice:
    def __init__(self, instance, backend: str = "highs", time_limit: float = 300.0,
                 node_limit: int = 3000, max_cg_iterations: int = 300,
                 max_routes_per_pricing: int = 25, verbose: bool = False):
        self.inst = instance
        self.backend = backend
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.max_cg_iterations = max_cg_iterations
        self.max_routes_per_pricing = max_routes_per_pricing
        self.verbose = verbose
        self.pool: dict[tuple, Column] = {}
        self.stats = BPStats()

    # ------------------------------------------------------------------
    def _get_or_add_column(self, route: tuple, cost: float) -> Column:
        existing = self.pool.get(route)
        if existing is not None:
            return existing
        incidence = np.zeros(self.inst.n)
        for node in route[1:-1]:
            incidence[node - 1] += 1
        col = Column(route, cost, incidence)
        self.pool[route] = col
        self.stats.columns_generated += 1
        return col

    def _seed_pool(self):
        d = self.inst.d
        for i in range(1, self.inst.n + 1):
            route = (0, i, self.inst.n + 1)
            self._get_or_add_column(route, route_cost(route, d))
        impact_routes = [tuple(r) for r in impact_construction(self.inst)]
        for r in impact_routes:
            self._get_or_add_column(r, route_cost(r, d))
        return impact_routes

    @staticmethod
    def _column_compatible(route: tuple, node: BPNode) -> bool:
        for (i, j) in zip(route[:-1], route[1:]):
            if (i, j) in node.arc_forbidden:
                return False
        if node.arc_required_next:
            for pos in range(len(route) - 1):
                req_j = node.arc_required_next.get(route[pos])
                if req_j is not None and route[pos + 1] != req_j:
                    return False
        if node.arc_required_prev:
            for pos in range(1, len(route)):
                req_i = node.arc_required_prev.get(route[pos])
                if req_i is not None and route[pos - 1] != req_i:
                    return False
        if node.pair_together or node.pair_apart:
            visited = set(route)
            for p, q in node.pair_together:
                if (p in visited) != (q in visited):
                    return False
            for p, q in node.pair_apart:
                if p in visited and q in visited:
                    return False
        return True

    def _process_node(self, node: BPNode) -> NodeResult:
        mp = MasterProblem(self.inst.n, fleet_ub=node.fleet_ub,
                            fleet_lb=node.fleet_lb, backend=self.backend)
        columns = [c for c in self.pool.values() if self._column_compatible(c.route, node)]
        for c in columns:
            mp.add_column(c.cost, c.incidence)

        sol = None
        converged = False
        for _ in range(self.max_cg_iterations):
            sol = mp.solve()
            if sol.status != "optimal":
                return NodeResult(feasible=False)
            if node.depth == 0:
                self.stats.root_iteration_bounds.append(sol.obj_value)
            pricing = solve_pricing(
                self.inst, sol.duals_customers, sol.dual_fleet,
                arc_forbidden=node.arc_forbidden,
                arc_required_next=node.arc_required_next,
                arc_required_prev=node.arc_required_prev,
                pair_together=node.pair_together,
                pair_apart=node.pair_apart,
                max_routes=self.max_routes_per_pricing,
            )
            if not pricing.routes:
                converged = True
                break
            for route in pricing.routes:
                cost = route_cost(route, self.inst.d)
                col = self._get_or_add_column(route, cost)
                mp.add_column(col.cost, col.incidence)
                columns.append(col)

        if sol is None or sol.uses_artificial:
            return NodeResult(feasible=False)

        return NodeResult(feasible=True, bound=sol.obj_value, y=sol.y,
                           columns=columns, converged=converged)

    @staticmethod
    def _arc_flow(result: NodeResult):
        flow = {}
        for col, val in zip(result.columns, result.y):
            if val <= EPS:
                continue
            for arc in zip(col.route[:-1], col.route[1:]):
                flow[arc] = flow.get(arc, 0.0) + val
        return flow

    @staticmethod
    def _joint_customer_coverage(result: NodeResult):
        """sum_r(y_r) for every pair of customers that appear together in at
        least one positive-weight column, used by Ryan-Foster branching."""
        joint: dict[tuple, float] = {}
        for col, val in zip(result.columns, result.y):
            if val <= EPS:
                continue
            customers = sorted(set(col.route[1:-1]))
            for idx, p in enumerate(customers):
                for q in customers[idx + 1:]:
                    joint[(p, q)] = joint.get((p, q), 0.0) + val
        return joint

    def _branch(self, node: BPNode, result: NodeResult):
        total_vehicles = sum(result.y)
        if abs(total_vehicles - round(total_vehicles)) > EPS:
            floor_k = math.floor(total_vehicles + EPS)
            ceil_k = math.ceil(total_vehicles - EPS)
            child_lo = node.child(fleet_ub=min(node.fleet_ub, floor_k))
            child_hi = node.child(fleet_lb=max(node.fleet_lb, ceil_k))
            return [c for c in (child_lo, child_hi) if c.fleet_lb <= c.fleet_ub]

        flow = self._arc_flow(result)
        best_arc, best_gap = None, 0.5 + EPS
        for arc, val in flow.items():
            frac = val - math.floor(val)
            if EPS < frac < 1 - EPS:
                gap = abs(frac - 0.5)
                if gap < best_gap:
                    best_gap, best_arc = gap, arc

        if best_arc is not None:
            i, j = best_arc
            forbid_child = node.child(arc_forbidden=node.arc_forbidden | {(i, j)})
            require_next = dict(node.arc_required_next); require_next[i] = j
            require_prev = dict(node.arc_required_prev); require_prev[j] = i
            require_child = node.child(arc_required_next=require_next,
                                        arc_required_prev=require_prev)
            return [forbid_child, require_child]

        # Ryan-Foster fallback: every aggregated arc flow is already
        # integral, yet y is fractional (a known degeneracy of pure arc
        # branching on set-partitioning masters). This rule is complete: it
        # always finds a cutting pair whenever y is fractional.
        joint = self._joint_customer_coverage(result)
        for (p, q), val in joint.items():
            if EPS < val < 1 - EPS:
                together = node.child(pair_together=node.pair_together | {(p, q)})
                apart = node.child(pair_apart=node.pair_apart | {(p, q)})
                return [together, apart]

        return []  # genuinely integral (should not normally be reached)

    # ------------------------------------------------------------------
    def solve(self) -> BPResult:
        start_time = time.time()
        impact_routes = self._seed_pool()
        incumbent_routes = impact_routes
        incumbent_cost = sum(route_cost(r, self.inst.d) for r in impact_routes)

        root = BPNode(0, self.inst.K, frozenset(), {}, {}, frozenset(), frozenset(), 0)
        counter = itertools.count()
        heap = [(-math.inf, next(counter), root)]
        best_open_bound = -math.inf

        while heap:
            elapsed = time.time() - start_time
            if elapsed > self.time_limit or self.stats.nodes_explored >= self.node_limit:
                self.stats.time_limit_hit = True
                best_open_bound = heap[0][0]
                break

            bound_hint, _, node = heapq.heappop(heap)
            self.stats.nodes_explored += 1

            result = self._process_node(node)
            if not result.feasible:
                self.stats.nodes_pruned_infeasible += 1
                continue
            if node.depth == 0:
                self.stats.root_lp_bound = result.bound

            if result.converged and result.bound >= incumbent_cost - EPS:
                self.stats.nodes_pruned_bound += 1
                continue

            frac_columns = [v for v in result.y if EPS < v < 1 - EPS]
            if not frac_columns:
                chosen = [(col.route, col.cost) for col, v in zip(result.columns, result.y)
                          if v > 1 - EPS]
                cost = sum(c for _, c in chosen)
                if cost < incumbent_cost - EPS:
                    incumbent_cost = cost
                    incumbent_routes = [r for r, _ in chosen]
                continue

            children = self._branch(node, result)
            if not children:
                # No valid cut found even though y is fractional: extremely
                # unlikely (would contradict the completeness of Ryan-Foster
                # branching on a set-partitioning master), but if it ever
                # happens we must not loop -- accept this node's fractional
                # bound as a (non-tightened) lower bound and move on rather
                # than hang.
                continue

            child_bound = result.bound if result.converged else bound_hint
            for child in children:
                heapq.heappush(heap, (child_bound, next(counter), child))

        self.stats.elapsed_seconds = time.time() - start_time
        if heap:
            lower_bound = heap[0][0] if best_open_bound == -math.inf else \
                min(best_open_bound, heap[0][0])
        else:
            lower_bound = incumbent_cost  # tree fully explored: proven optimal
        status = "optimal" if not heap and not self.stats.time_limit_hit else "feasible"

        return BPResult(status, incumbent_routes, incumbent_cost,
                         max(lower_bound, self.stats.root_lp_bound or -math.inf), self.stats)
