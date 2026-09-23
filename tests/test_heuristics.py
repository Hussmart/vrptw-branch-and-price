from vrptw_cg.data import load_instance, route_cost
from vrptw_cg.heuristics import impact_construction


def test_impact_construction_covers_every_customer_exactly_once():
    inst = load_instance("c101", 15)
    routes = impact_construction(inst)
    covered = []
    for route in routes:
        assert route[0] == 0 and route[-1] == inst.n + 1
        covered.extend(route[1:-1])
    assert sorted(covered) == list(range(1, inst.n + 1))


def test_impact_construction_routes_are_feasible():
    inst = load_instance("r101", 15)
    routes = impact_construction(inst)
    for route in routes:
        assert sum(inst.q[c] for c in route[1:-1]) <= inst.Q + 1e-6
        t = inst.a[0]
        for a, b in zip(route[:-1], route[1:]):
            t = max(t + inst.d[a, b], inst.a[b])
            assert t <= inst.b[b] + 1e-6


def test_impact_construction_uses_at_most_available_fleet_in_practice():
    inst = load_instance("c101", 25)
    routes = impact_construction(inst)
    # Not a hard constraint of the heuristic itself, but a sanity check that
    # it does not produce a degenerate one-customer-per-route solution.
    assert len(routes) < inst.n
    total_cost = sum(route_cost(r, inst.d) for r in routes)
    assert total_cost > 0
