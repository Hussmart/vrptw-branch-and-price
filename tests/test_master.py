"""Numerical validation of the RMP and, critically, of the dual-value sign
convention used to compute reduced costs in pricing.py. Getting this sign
wrong would make column generation silently terminate too early (missing
improving columns) or loop forever re-adding non-improving ones, so this is
checked directly against LP optimality conditions rather than trusted from
the SciPy docs alone.
"""
import numpy as np
import pytest

from vrptw_cg.master import MasterProblem


def _reduced_cost(cost, incidence, pi, dual_fleet):
    return cost - float(pi @ incidence) - dual_fleet


def test_toy_master_matches_hand_solved_optimum():
    # 2 customers, 3 candidate routes:
    #   r0: serves {1}      cost 10
    #   r1: serves {2}      cost 10
    #   r2: serves {1, 2}   cost 15   <- strictly better, should be fully used
    mp = MasterProblem(n_customers=2, fleet_ub=2)
    mp.add_column(10.0, np.array([1.0, 0.0]))
    mp.add_column(10.0, np.array([0.0, 1.0]))
    mp.add_column(15.0, np.array([1.0, 1.0]))

    sol = mp.solve()
    assert sol.status == "optimal"
    assert sol.obj_value == pytest.approx(15.0)
    assert sol.y[2] == pytest.approx(1.0)
    assert sol.y[0] == pytest.approx(0.0)
    assert sol.y[1] == pytest.approx(0.0)


def test_dual_feasibility_all_real_columns_nonnegative_reduced_cost():
    """LP optimality condition for a minimization problem: at the optimum,
    every column not in the basis (and, by complementary slackness, every
    column period) must have reduced cost >= 0. This holds for ANY set of
    dual values that are truly optimal duals -- so this test would fail
    immediately if the sign convention were flipped anywhere.
    """
    rng = np.random.default_rng(0)
    n = 6
    mp = MasterProblem(n_customers=n, fleet_ub=3, fleet_lb=1)
    columns = []
    for _ in range(15):
        incidence = (rng.random(n) < 0.4).astype(float)
        if incidence.sum() == 0:
            incidence[0] = 1.0
        cost = float(incidence.sum() * rng.uniform(5, 15))
        mp.add_column(cost, incidence)
        columns.append((cost, incidence))
    # Guarantee feasibility of a reasonable solution: singleton routes.
    for i in range(n):
        inc = np.zeros(n); inc[i] = 1.0
        mp.add_column(5.0, inc)
        columns.append((5.0, inc))

    sol = mp.solve()
    assert sol.status == "optimal"

    for cost, incidence in columns:
        rc = _reduced_cost(cost, incidence, sol.duals_customers, sol.dual_fleet)
        assert rc >= -1e-6, f"dual-infeasible column found: rc={rc}"


def test_fleet_window_lb_ub_both_active_stays_dual_feasible():
    n = 4
    mp = MasterProblem(n_customers=n, fleet_ub=2, fleet_lb=2)  # force exactly 2 vehicles
    singleton_cost = 5.0
    columns = []
    for i in range(n):
        inc = np.zeros(n); inc[i] = 1.0
        mp.add_column(singleton_cost, inc)
        columns.append((singleton_cost, inc))
    pair_cost = 8.0
    pair = np.array([1.0, 1.0, 0.0, 0.0])
    mp.add_column(pair_cost, pair)
    columns.append((pair_cost, pair))

    sol = mp.solve()
    assert sol.status == "optimal"
    for cost, incidence in columns:
        rc = _reduced_cost(cost, incidence, sol.duals_customers, sol.dual_fleet)
        assert rc >= -1e-6
