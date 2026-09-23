"""End-to-end correctness check: an independent bitmask-DP brute-force exact
solver (unrelated to column generation) is used as ground truth against the
Branch-and-Price result on tiny instances. Agreement between two completely
different algorithms is much stronger evidence of correctness than either
one's internal consistency -- this is the test to point to when defending
the implementation.
"""
from itertools import permutations

import pytest

from vrptw_cg.branch_and_price import BranchAndPrice
from vrptw_cg.data import load_instance


def _feasible_route_cost(subset, inst):
    """Cheapest feasible permutation of `subset` as a single route, or None
    if no permutation respects capacity and time windows."""
    if sum(inst.q[c] for c in subset) > inst.Q + 1e-9:
        return None
    best = None
    n1 = inst.n + 1
    for perm in permutations(subset):
        t = inst.a[0]
        ok = True
        cost = 0.0
        prev = 0
        for c in perm:
            cost += inst.d[prev, c]
            t = max(t + inst.d[prev, c], inst.a[c])
            if t > inst.b[c] + 1e-9:
                ok = False
                break
            prev = c
        if not ok:
            continue
        cost += inst.d[prev, n1]
        t = max(t + inst.d[prev, n1], inst.a[n1])
        if t > inst.b[n1] + 1e-9:
            continue
        if best is None or cost < best:
            best = cost
    return best


def bruteforce_optimal(inst):
    """Exact bitmask-DP set-partitioning solve: ground truth, independent of
    column generation / branch-and-price machinery entirely."""
    n = inst.n
    route_cost_of = {}
    for mask in range(1, 1 << n):
        subset = tuple(i + 1 for i in range(n) if mask & (1 << i))
        if len(subset) > 6:
            continue  # keep brute force tractable; tests use n <= 6
        cost = _feasible_route_cost(subset, inst)
        if cost is not None:
            route_cost_of[mask] = cost

    full = (1 << n) - 1
    INF = float("inf")
    # dp[mask][v] = min cost to cover `mask` using exactly v routes
    dp = [[INF] * (inst.K + 1) for _ in range(1 << n)]
    dp[0][0] = 0.0
    for mask in range(1 << n):
        for v in range(inst.K):
            if dp[mask][v] == INF:
                continue
            remaining = full & ~mask
            sub = remaining
            while sub:
                if sub in route_cost_of:
                    new_mask = mask | sub
                    cost = dp[mask][v] + route_cost_of[sub]
                    if cost < dp[new_mask][v + 1]:
                        dp[new_mask][v + 1] = cost
                sub = (sub - 1) & remaining
    best = min(dp[full][v] for v in range(inst.K + 1))
    return best


@pytest.mark.parametrize("instance_name,n", [("c101", 5), ("r101", 5), ("rc101", 6)])
def test_branch_and_price_matches_bruteforce_oracle(instance_name, n):
    inst = load_instance(instance_name, n)
    oracle_cost = bruteforce_optimal(inst)
    assert oracle_cost < float("inf"), "instance should be feasible"

    solver = BranchAndPrice(inst, time_limit=60.0, node_limit=500)
    result = solver.solve()

    assert result.status == "optimal"
    assert result.incumbent_cost == pytest.approx(oracle_cost, abs=1e-3)


def test_solution_covers_every_customer_exactly_once():
    inst = load_instance("c101", 10)
    result = BranchAndPrice(inst, time_limit=60.0, node_limit=1000).solve()
    covered = []
    for route in result.incumbent_routes:
        covered.extend(route[1:-1])
    assert sorted(covered) == list(range(1, inst.n + 1))


def test_solution_respects_capacity_and_time_windows():
    inst = load_instance("r101", 10)
    result = BranchAndPrice(inst, time_limit=60.0, node_limit=1000).solve()
    for route in result.incumbent_routes:
        assert sum(inst.q[c] for c in route[1:-1]) <= inst.Q + 1e-6
        t = inst.a[0]
        for a, b in zip(route[:-1], route[1:]):
            t = max(t + inst.d[a, b], inst.a[b])
            assert t <= inst.b[b] + 1e-6


def test_lower_bound_never_exceeds_incumbent():
    inst = load_instance("rc101", 10)
    result = BranchAndPrice(inst, time_limit=30.0, node_limit=300).solve()
    assert result.lower_bound <= result.incumbent_cost + 1e-6
