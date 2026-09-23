import numpy as np

from vrptw_cg.data import load_instance
from vrptw_cg.pricing import solve_pricing


def _all_duals_zero(inst):
    return np.zeros(inst.n)


def test_pricing_finds_negative_reduced_cost_when_duals_are_generous():
    inst = load_instance("c101", 6, neighbor_k=5)
    # Duals equal to each customer's cheapest incident arc: guarantees some
    # route has strictly negative reduced cost (any 2+-customer route beats
    # the sum of "give every customer its own dual").
    duals = np.array([min(inst.d[0, i], inst.d[i, inst.n + 1]) for i in range(1, inst.n + 1)])
    result = solve_pricing(inst, duals, dual_fleet=0.0)
    assert result.routes, "expected at least one improving column"
    for rc in result.reduced_costs:
        assert rc < 0


def test_pricing_routes_are_capacity_and_time_feasible():
    inst = load_instance("r101", 10, neighbor_k=6)
    duals = np.full(inst.n, 5.0)
    result = solve_pricing(inst, duals, dual_fleet=0.0, max_routes=50)
    for route in result.routes:
        load = sum(inst.q[node] for node in route[1:-1])
        assert load <= inst.Q + 1e-6
        t = inst.a[0]
        for a, b in zip(route[:-1], route[1:]):
            t = max(t + inst.d[a, b], inst.a[b])
            assert t <= inst.b[b] + 1e-6


def test_pricing_routes_never_repeat_a_customer():
    inst = load_instance("rc101", 10, neighbor_k=6)
    duals = np.full(inst.n, 5.0)
    result = solve_pricing(inst, duals, dual_fleet=0.0, max_routes=50)
    for route in result.routes:
        customers = route[1:-1]
        assert len(customers) == len(set(customers))


def test_arc_forbidden_is_respected():
    inst = load_instance("c101", 6, neighbor_k=5)
    duals = np.array([min(inst.d[0, i], inst.d[i, inst.n + 1]) for i in range(1, inst.n + 1)])
    baseline = solve_pricing(inst, duals, dual_fleet=0.0, max_routes=50)
    assert baseline.routes
    arc = (baseline.routes[0][0], baseline.routes[0][1])

    restricted = solve_pricing(inst, duals, dual_fleet=0.0, max_routes=50,
                                arc_forbidden={arc})
    for route in restricted.routes:
        for a, b in zip(route[:-1], route[1:]):
            assert (a, b) != arc


def test_arc_required_next_is_respected():
    inst = load_instance("c101", 6, neighbor_k=5)
    duals = np.array([min(inst.d[0, i], inst.d[i, inst.n + 1]) for i in range(1, inst.n + 1)])
    result = solve_pricing(inst, duals, dual_fleet=0.0, max_routes=50,
                            arc_required_next={0: 1})
    for route in result.routes:
        if 0 in route[:-1]:
            pos = route.index(0)
            assert route[pos + 1] == 1


def test_pair_together_and_apart_are_respected():
    inst = load_instance("c101", 8, neighbor_k=6)
    duals = np.full(inst.n, 1.0)
    together = solve_pricing(inst, duals, dual_fleet=0.0, max_routes=100,
                              pair_together=frozenset({(1, 2)}))
    for route in together.routes:
        visited = set(route)
        assert (1 in visited) == (2 in visited)

    apart = solve_pricing(inst, duals, dual_fleet=0.0, max_routes=100,
                           pair_apart=frozenset({(1, 2)}))
    for route in apart.routes:
        visited = set(route)
        assert not (1 in visited and 2 in visited)
