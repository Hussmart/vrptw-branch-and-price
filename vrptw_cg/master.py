"""Restricted Master Problem (RMP): the set-partitioning LP relaxation.

    minimize    sum_r  cost_r * y_r  +  BIG_M * sum_i z_i
    subject to  sum_r  a_{i,r} * y_r  +  z_i  =  1        for every customer i
                fleet_lb  <=  sum_r  y_r  <=  fleet_ub     (vehicle count window)
                y_r, z_i >= 0

``z_i`` are per-customer artificial variables with a very large cost. They
guarantee the RMP is *always* feasible, even deep in the branch-and-price
tree where arc branching may have removed every real route touching some
customer for the columns generated so far -- a standard requirement for a
correct B&P implementation (see e.g. Vanderbeck & Wolsey, 1996). A node
whose LP optimum still uses an artificial variable is genuinely infeasible
under its branching restrictions and is pruned by the caller.

``fleet_lb``/``fleet_ub`` form a *window* on the number of vehicles used
instead of a single fixed fleet size, because the branch-and-price tree
branches on the (possibly fractional) total vehicle count sum(y_r) before it
branches on individual arcs -- exactly the two-level scheme of Desrochers,
Desrosiers & Solomon (1992), the paper this project was originally built
from.

The original project solved this with Gurobi only, which needs a paid
license. The default backend here is SciPy's ``linprog`` with the HiGHS
method, which is free, open source, and ships with SciPy itself -- no
separate installation and no license file. Gurobi remains available as an
optional backend for the fixed-fleet-size case (see the docstring of
``_solve_gurobi``) for users who already have a license.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack


@dataclass
class MasterSolution:
    status: str                          # "optimal", "infeasible"
    obj_value: float | None
    y: np.ndarray | None                 # primal values, one per real column
    duals_customers: np.ndarray | None   # pi_i, one per customer
    dual_fleet: float | None             # net reduced-cost contribution per route
    uses_artificial: bool = False        # True => node infeasible for real routes


class MasterProblem:
    """Thin backend-agnostic wrapper around the RMP LP."""

    def __init__(self, n_customers: int, fleet_ub: int, fleet_lb: int = 0,
                 backend: str = "highs", big_m: float | None = None):
        self.n = n_customers
        self.fleet_ub = fleet_ub
        self.fleet_lb = fleet_lb
        self.backend = backend
        self.big_m = big_m if big_m is not None else 1e6
        self.costs: list[float] = []
        self.columns: list[np.ndarray] = []   # each: 0/1 incidence vector, length n

    def add_column(self, cost: float, incidence: np.ndarray) -> int:
        self.costs.append(cost)
        self.columns.append(incidence)
        return len(self.costs) - 1

    def num_columns(self) -> int:
        return len(self.costs)

    def solve(self) -> MasterSolution:
        if self.backend == "gurobi":
            return self._solve_gurobi()
        return self._solve_highs()

    # ------------------------------------------------------------------
    def _real_arrays(self):
        c = np.array(self.costs, dtype=float)
        A = np.column_stack(self.columns) if self.columns else \
            np.zeros((self.n, 0))
        return c, A

    def _solve_highs(self) -> MasterSolution:
        c_real, A_real = self._real_arrays()
        n_real = A_real.shape[1]

        c = np.concatenate([c_real, np.full(self.n, self.big_m)])
        A_eq = hstack([csr_matrix(A_real), identity(self.n, format="csr")])
        b_eq = np.ones(self.n)

        # Fleet-size window as up to two inequality rows. Coefficients are
        # entered exactly as passed to the solver (+1 for the upper-bound
        # row, -1 for the lower-bound row, since SciPy only accepts `<=`),
        # so the generic reduced-cost formula rc_j = c_j - sum(dual_i * A_ij)
        # can be applied uniformly with the *solver-reported* marginals and
        # the *as-entered* coefficients -- see tests/test_master.py for a
        # numerical check of dual feasibility (all real reduced costs >= 0
        # at the reported optimum).
        rows, rhs, row_coef = [], [], []
        if self.fleet_ub is not None:
            rows.append(np.concatenate([np.ones(n_real), np.zeros(self.n)]))
            rhs.append(float(self.fleet_ub))
            row_coef.append(1.0)
        if self.fleet_lb and self.fleet_lb > 0:
            rows.append(np.concatenate([-np.ones(n_real), np.zeros(self.n)]))
            rhs.append(-float(self.fleet_lb))
            row_coef.append(-1.0)

        if rows:
            A_ub = csr_matrix(np.vstack(rows))
            b_ub = np.array(rhs)
        else:
            A_ub = b_ub = None

        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                       bounds=(0, None), method="highs")

        if not res.success:
            return MasterSolution("infeasible", None, None, None, None)

        # SciPy's `marginals` are shadow prices d(obj)/d(rhs), i.e. exactly
        # the LP dual variables used in the reduced-cost formula -- no sign
        # flip needed. Verified empirically against hand-solved toy LPs; see
        # docs/REPORT.md, "Dual sign convention" note.
        pi = np.asarray(res.eqlin.marginals)
        # Generic LP reduced-cost identity: rc_j = c_j - sum_i(dual_i * A_ij).
        # dual_fleet below IS that sum (over the one or two fleet rows) using
        # the coefficients exactly as entered into the solver (row_coef) and
        # the solver-reported marginals -- solve_pricing() subtracts it once
        # per route, matching this identity exactly.
        dual_fleet = 0.0
        if rows:
            marginals = np.atleast_1d(res.ineqlin.marginals)
            dual_fleet = float(sum(m * coef for m, coef in zip(marginals, row_coef)))

        y = res.x[:n_real]
        z = res.x[n_real:]
        real_obj = float(c_real @ y) if n_real else 0.0
        return MasterSolution("optimal", real_obj, y, pi, dual_fleet,
                               uses_artificial=bool(np.any(z > 1e-6)))

    def _solve_gurobi(self) -> MasterSolution:
        """Optional Gurobi backend. Only supports a fixed fleet size
        (fleet_lb must be 0) -- the sign convention for Gurobi's dual value
        on a native '>=' constraint has not been empirically verified in
        this project (no Gurobi license available during development), so
        vehicle-count branching is restricted to the HiGHS backend, which
        *is* verified (tests/test_master.py).
        """
        if self.fleet_lb:
            raise NotImplementedError(
                "backend='gurobi' does not support fleet_lb branching; "
                "use the default backend='highs' for full branch-and-price."
            )
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError(
                "backend='gurobi' requires gurobipy and a valid license. "
                "Use backend='highs' (default) for a license-free solve."
            ) from exc

        c_real, A_real = self._real_arrays()
        n_real = A_real.shape[1]

        model = gp.Model("RMP")
        model.Params.OutputFlag = 0
        y = model.addMVar(shape=n_real, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
        z = model.addMVar(shape=self.n, vtype=GRB.CONTINUOUS, lb=0.0, name="z")
        model.setObjective(c_real @ y + self.big_m * z.sum(), GRB.MINIMIZE)
        cover = model.addConstr(A_real @ y + z == np.ones(self.n), name="cover")
        fleet = model.addConstr(y.sum() <= self.fleet_ub, name="fleet")
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            return MasterSolution("infeasible", None, None, None, None)

        pi = np.array([cover[i].Pi for i in range(self.n)])
        dual_fleet = fleet.Pi          # coefficient +1, same identity as HiGHS path
        real_obj = float(c_real @ y.X) if n_real else 0.0
        return MasterSolution("optimal", real_obj, y.X, pi, dual_fleet,
                               uses_artificial=bool(np.any(z.X > 1e-6)))
