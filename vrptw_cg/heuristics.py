"""Constructive and rounding heuristics.

``impact_construction`` builds an initial feasible solution with the IMPACT
insertion heuristic (Ioannou, Kritikos & Prastacos, 2001), used only to seed
the root node of the branch-and-price tree with a good starting column pool
and an initial incumbent upper bound -- it plays no role in the exactness of
the final answer.

``greedy_set_cover`` is the same idea as the original project's
``coverCost.py``: given a pool of already-generated columns, greedily pick
the best coverage/cost ratio route until every customer is covered. It is
used as a fast fallback incumbent, not as the reported "solution" (that role
is filled by the branch-and-price integer optimum).
"""
from __future__ import annotations

from .data import route_cost

B1 = B2 = B3 = 1.0 / 3.0


def _insert_node(route, node, position, s, arr, d, a):
    new_route = route[:position] + [node] + route[position:]
    new_s = s[:position]
    new_arr = arr[:position]
    for i in range(position, len(new_route)):
        prev_s = new_s[i - 1]
        new_arr.append(prev_s + d[new_route[i - 1], new_route[i]])
        new_s.append(max(new_arr[i], a[new_route[i]]))
    return new_route, new_s, new_arr


def _route_is_feasible(route, a, b, s, q, Q):
    if sum(q[node] for node in route) > Q:
        return False
    return all(a[route[i]] - 1e-9 <= s[i] <= b[route[i]] + 1e-9
               for i in range(len(route)))


def _compute_is_iu_ld(pos_u, route, arr, s, a, b, d, j_minus_u):
    u = route[pos_u]
    i, j = route[pos_u - 1], route[pos_u + 1]
    IS = arr[pos_u] - a[u]
    if j_minus_u:
        IU = sum(max(b[nb] - a[u] - d[u, nb], b[u] - a[nb] - d[u, nb])
                  for nb in j_minus_u) / len(j_minus_u)
    else:
        IU = 0.0
    c1 = d[i, u] + d[u, j] - d[i, j]
    c2 = (b[j] - (arr[pos_u - 1] + d[i, j])) - (b[j] - (arr[pos_u] + d[i, j]))
    c3 = b[u] - (arr[pos_u - 1] + d[i, u])
    LD = B1 * c1 + B2 * c2 + B3 * c3
    return IS, IU, LD


def impact_construction(instance):
    """Return a list of routes (each a list of node indices, 0 ... n+1)."""
    n, d, a, b, q, Q = (instance.n, instance.d, instance.a, instance.b,
                         instance.q, instance.Q)
    remaining = list(range(1, n + 1))
    routes = []

    while remaining:
        far = max(remaining, key=lambda j: d[0, j])
        route = [0, far, n + 1]
        arr = [0.0, d[0, far]]
        s = [0.0, max(a[far], arr[1])]
        arr.append(s[1] + d[far, n + 1])
        s.append(max(arr[2], a[n + 1]))
        remaining.remove(far)

        feasible = remaining[:]
        while feasible:
            proposals = {}
            for u in feasible:
                j_minus_u = [c for c in remaining if c != u]
                best_impact, best_pos = None, None
                is_list, iu_list, ld_list, positions = [], [], [], []
                for pos in range(1, len(route)):
                    new_route, new_s, new_arr = _insert_node(route, u, pos, s, arr, d, a)
                    if _route_is_feasible(new_route, a, b, new_s, q, Q):
                        positions.append(pos)
                        Is, Iu, Ld = _compute_is_iu_ld(pos, new_route, new_arr,
                                                        new_s, a, b, d, j_minus_u)
                        is_list.append(Is); iu_list.append(Iu); ld_list.append(Ld)
                if not positions:
                    feasible.remove(u)
                    continue
                ir = sum(ld_list) / len(positions)
                for k, pos in enumerate(positions):
                    impact = is_list[k] + iu_list[k] + ir
                    if best_impact is None or impact < best_impact:
                        best_impact, best_pos = impact, pos
                proposals[best_impact] = (u, best_pos)

            if not proposals:
                break
            node_to_insert, insert_pos = proposals[min(proposals)]
            route, s, arr = _insert_node(route, node_to_insert, insert_pos, s, arr, d, a)
            feasible.remove(node_to_insert)
            remaining.remove(node_to_insert)

        routes.append(route)

    return routes


def greedy_set_cover(instance, routes: list[tuple]):
    """Greedy coverage/cost rounding over an already-generated column pool.
    Returns (solution_routes, total_cost) or None if no cover is found.
    """
    d = instance.d
    n = instance.n
    costs = [route_cost(r, d) for r in routes]
    cover_ratio = [(len(r) - 2) / c if c > 0 else 0.0 for r, c in zip(routes, costs)]

    best_solution, best_cost = None, float("inf")
    order = sorted(range(len(routes)), key=lambda i: -cover_ratio[i])
    for start_idx in order[:min(len(order), 25)]:
        remaining = set(range(1, n + 1))
        solution, cost = [], 0.0
        pool = list(range(len(routes)))
        idx = start_idx
        while True:
            route = routes[idx]
            solution.append(route)
            cost += costs[idx]
            for node in route[1:-1]:
                remaining.discard(node)
            if not remaining:
                break
            candidates = [i for i in pool
                          if all(node in remaining for node in routes[i][1:-1])
                          and any(node in remaining for node in routes[i][1:-1])]
            if not candidates:
                solution = None
                break
            idx = max(candidates, key=lambda i: cover_ratio[i])
        if solution is not None and cost < best_cost:
            best_solution, best_cost = solution, cost

    if best_solution is None:
        return None
    return best_solution, best_cost
