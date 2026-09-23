# Branch-and-Price for the VRPTW: formulation, algorithms, and validation

## 1. Problem

The Vehicle Routing Problem with Time Windows (VRPTW) asks for a minimum-cost
set of routes, each starting and ending at a depot, that together visit every
customer exactly once, subject to:

* **Capacity**: the total demand served by a route cannot exceed vehicle
  capacity `Q`.
* **Time windows**: a vehicle must arrive at customer `i` within
  `[a_i, b_i]`; arriving early means waiting until `a_i`.
* **Fleet size**: at most `K` vehicles are available.

Node `0` and node `n+1` both represent the depot (route start and end copies
of the same physical location), so a route is a path `0 -> i_1 -> ... ->
i_k -> n+1`. This convention, and the Solomon instance format, are kept
identical to the codebase this project forked (see `NOTICE.md`).

## 2. Dantzig-Wolfe decomposition

Enumerating every feasible route explicitly is intractable, so the problem
is decomposed (Desrochers, Desrosiers & Solomon, 1992) into:

**Master problem** -- pick a minimum-cost combination of routes covering
every customer exactly once:

```
minimize    sum_r  cost_r * y_r
subject to  sum_r  a_{i,r} * y_r  =  1      for every customer i   [dual pi_i]
            sum_r  y_r            <= K      fleet size             [dual sigma]
            y_r in {0, 1}
```

Relaxing `y_r in {0,1}` to `y_r >= 0` gives the LP relaxation solved by
column generation. Its optimal value is a valid lower bound on the true
integer optimum.

**Pricing subproblem** -- given the master's dual prices, find routes with
negative *reduced cost*

```
rc(r) = cost_r - sum_{i in r} pi_i - sigma
```

This is an Elementary Shortest Path Problem with Resource Constraints
(ESPPRC): shortest path from node 0 to node n+1 under capacity and time-window
constraints, without repeating a customer. ESPPRC is NP-hard in general.

### Why the original repository's version of this doesn't work

The forked repository (`legacy/`) solved a **non-elementary** relaxation of
the pricing problem (allowing customer repeats) via a resource-indexed DP,
and separately attempted the **exact** elementary version as a MIP
(`legacy/ESPmodel.py`) -- which times out after ~160 minutes even on small
instances (`legacy/note.txt`). Neither is what current branch-and-price
codes use in practice.

## 3. ng-route relaxation pricing (`vrptw_cg/pricing.py`)

This project uses the **ng-route relaxation** (Baldacci, Mingozzi & Roberti,
2011), the standard middle ground used in essentially every modern VRP
branch-cut-and-price code (BaPCod/VRPSolver, RouteOpt, the bucket-graph
labeling algorithm of Sadykov, Uchoa & Pessoa, 2021):

* Each customer `j` has a fixed neighbor set `N(j)`, the `k` nearest
  customers (default `k=8`).
* A label carries a *memory* set `M subset of N(current node)` instead of
  the full visited-set.
* Extending a label from `i` to `j` is forbidden if `j` is already in the
  label's memory -- this blocks short, "obviously bad" cycles cheaply.
* The new memory at `j` is `(M intersect N(j)) union {j}`.

This keeps the label state space polynomially bounded (unlike full
elementary enforcement) while remaining far tighter than no elementarity
constraint at all, and it is exact whenever the LP/IP optimal routes happen
to avoid cycles outside the neighbor structure -- which the branch-and-price
layer (Section 4) verifies and corrects for regardless.

Labels are extended with a standard forward label-setting algorithm with
dominance pruning: label `L1` dominates `L2` at the same node if it is
equal-or-better on cost, load, and time, and its memory set is a subset of
`L2`'s (a smaller memory is strictly easier to extend from). Branching
restrictions (arc forbid/require, Ryan-Foster pairs; Section 4) are enforced
during extension or as a cheap post-filter on completed routes.

## 4. Branch-and-Price (`vrptw_cg/branch_and_price.py`)

Column generation alone only produces a lower bound. To certify an integer
optimum, every node of a branch-and-bound tree re-runs column generation
under extra restrictions (Barnhart et al., 1998). Three branching rules are
used, applied in this priority order, following Desrochers-Desrosiers-Solomon
(1992) plus the completeness fix from Ryan & Foster (1981):

1. **Vehicle-count branching**: if `sum(y_r)` is fractional, split into
   `sum(y_r) <= floor(k)` and `sum(y_r) >= ceil(k)` by tightening the
   master's fleet-size window.
2. **Arc branching**: otherwise, take the aggregated arc flow
   `x_ij = sum_r (y_r : arc (i,j) in r)` closest to 0.5 and branch on
   forbidding / requiring that arc (enforced directly inside the pricing
   DP, and as a filter on inherited columns).
3. **Ryan-Foster branching**: arc branching is *not* a complete branching
   rule for a set-partitioning master -- there exist fractional solutions
   where every aggregated arc flow is already integral (e.g. two distinct
   routes that overlap perfectly on every arc's total flow while each
   individual `y_r` stays fractional). When step 2 finds no fractional arc,
   the code falls back to picking two customers whose *joint* coverage
   `sum_r(y_r : both in r)` is fractional and branches on "must share a
   route" vs. "must never share a route". This rule is provably complete for
   set-partitioning formulations, which is what actually guarantees
   termination.

**Node feasibility.** Every master LP includes one artificial variable per
customer with a very large cost (`vrptw_cg/master.py`), so the LP is always
feasible; a node is pruned as *routing*-infeasible only when its converged
solution still relies on an artificial variable (Vanderbeck & Wolsey, 1996).

**Search strategy.** Best-first: nodes are explored in order of their
(converged, when available) LP bound, using a global incumbent for bound
pruning. The reported `lower_bound` is the minimum bound among unexplored
nodes when a time/node budget is hit, giving an honest optimality gap
instead of silently reporting a heuristic number as exact.

## 5. Two objectives: distance-only vs. the standard lexicographic objective

The original repository minimized distance only, and (as noted above) never
even enforced the fleet-size bound it accepted as a parameter. Published
Solomon-benchmark result tables instead use a **lexicographic** objective:
minimize the number of vehicles first, then minimize distance among
solutions that use that minimum fleet (Solomon, 1987). Comparing a
distance-only number against a lexicographic-optimal number is an
apples-to-oranges mistake that is easy to make silently, so this project
implements both explicitly and names them:

* `objective="distance"` (`BranchAndPrice` default): minimize distance
  subject to `sum(y_r) <= K`.
* `objective="count"` + `solve_lexicographic()`: phase 1 minimizes the
  number of vehicles by giving every real column a cost of exactly 1
  regardless of its length (`vrptw_cg/pricing.py`'s `arc_cost` parameter
  decouples the reduced-cost bookkeeping from the real distance matrix used
  for capacity/time-window feasibility, so the same label-setting algorithm
  serves both objectives). Phase 2 then re-solves from scratch with the
  fleet size capped at the phase-1 optimum `K*` and the normal distance
  objective. Both phases are exact Branch-and-Price runs, not heuristics.

`test_lexicographic_objective_matches_bruteforce_oracle` extends the
bitmask-DP oracle (Section 8) to also compute the true lexicographic optimum
and checks `solve_lexicographic()` against it on small instances.

## 6. Column generation acceleration: dual stabilization

Plain (unstabilized) column generation is well known for "tailing-off": many
late-stage iterations that each improve the LP bound by a negligible amount,
because the dual values oscillate as they approach optimality. This project
implements dual-value smoothing (du Merle, Villeneuve, Desrosiers & Hansen,
"Stabilized column generation", *Discrete Mathematics*, 1999):

1. Maintain a stability center `pi_bar` (initialized to the first duals
   seen at a node).
2. At each iteration, price with a blend `alpha * pi_bar + (1-alpha) * pi`
   of the center and the current exact duals (`stabilization_alpha`,
   default 0.5).
3. **If pricing with the blended duals finds no improving column, this is
   not a valid convergence certificate** -- only the true current duals are
   the actual dual-optimal solution of the current RMP. The exact duals are
   always re-tried before the node is allowed to declare convergence.
4. The center is moved towards whichever duals just succeeded.

Step 3 is what makes this purely an acceleration: it can only ever add one
extra pricing call per iteration in the worst case (when the blend fails and
the exact duals are needed anyway), never change which columns are
considered valid or which node bound is accepted.
`test_stabilization_does_not_change_the_optimal_answer` runs the same
instances with `stabilization_alpha=0` and `0.7` and checks the certified
optimum is bit-for-bit the same.

## 7. Correctness-critical implementation details

**Dual sign convention.** SciPy's `linprog(method="highs")` reports
`res.eqlin.marginals` / `res.ineqlin.marginals`. These were verified
empirically against hand-solved toy LPs (`min 2x1+3x2 s.t. x1+x2=1`, and a
binding `<=` case) to equal the shadow price `d(objective)/d(rhs)` directly,
with **no sign flip**, before being trusted inside the pricing reduced-cost
formula. This matters because a flipped sign would not raise an exception --
it would make column generation either terminate immediately (missing real
improving columns) or loop forever re-adding non-improving ones, silently
producing a wrong "optimal" answer. See `tests/test_master.py`.

**Fleet-size window.** Vehicle-count branching needs both a lower and an
upper bound on `sum(y_r)`, entered into `linprog` as two separate `<=` rows
(SciPy has no native `>=`). The reduced-cost contribution of each row is
computed generically as `-marginal * coefficient-as-entered`, which is
robust to the row transformation; `tests/test_master.py` checks dual
feasibility (every column's reduced cost `>= 0` at the optimum) with both
rows simultaneously active.

## 8. Validation

Three independent checks, beyond ordinary unit tests of individual functions:

1. **Dual feasibility** (`tests/test_master.py`): for random and
   hand-constructed column sets, every column's reduced cost is `>= -1e-6`
   at the reported LP optimum -- the fundamental LP optimality condition,
   checked numerically rather than assumed from documentation.
2. **Independent brute-force oracle** (`tests/test_branch_and_price.py`): a
   bitmask dynamic program over feasible-route subsets -- an algorithm that
   shares no code with column generation or label-setting -- computes both
   the plain optimum and the lexicographic (vehicles, then distance) optimum
   for small instances (`n <= 6`). Branch-and-Price and `solve_lexicographic`
   match it exactly on every tested Solomon instance class (clustered
   `c101`, random `r101`, mixed `rc101`).
3. **Stabilization non-interference**: the same instances solved with
   `stabilization_alpha=0` and `0.7` reach the identical certified optimum,
   confirming dual smoothing (Section 6) is a pure acceleration.

Run `pytest` to reproduce all three (30+ tests, ~1-2 minutes).

## 9. Computational results

Generated with `python -m vrptw_cg.benchmark --instances c101 c201 r101 r201
rc101 rc201 --customer-counts 10 25 --time-limit 60`, HiGHS backend, default
`k=8` ng-route neighborhoods, on the machine used for development (no
special hardware). Full machine-readable output: `results/benchmark.csv`.

| instance | customers | status | cost | lower bound | gap | vehicles | nodes explored | columns generated | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| c101 | 10 | optimal | 59.00 | 59.00 | 0.000% | 1 | 1 | 149 | 0.2 |
| c101 | 25 | optimal | 192.00 | 192.00 | 0.000% | 3 | 1 | 2144 | 26.3 |
| c201 | 10 | optimal | 153.00 | 153.00 | 0.000% | 2 | 1 | 113 | 0.2 |
| c201 | 25 | optimal | 217.00 | 217.00 | 0.000% | 2 | 1 | 2586 | 24.4 |
| r101 | 10 | optimal | 253.00 | 253.00 | 0.000% | 3 | 1 | 41 | 0.0 |
| r101 | 25 | optimal | 580.00 | 580.00 | 0.000% | 6 | 5 | 284 | 0.9 |
| r201 | 10 | optimal | 253.00 | 253.00 | 0.000% | 2 | 7 | 267 | 0.6 |
| r201 | 25 | optimal | 462.00 | 462.00 | 0.000% | 4 | 3 | 931 | 34.0 |
| rc101 | 10 | optimal | 184.00 | 184.00 | 0.000% | 2 | 1 | 148 | 0.4 |
| rc101 | 25 | optimal | 356.00 | 356.00 | 0.000% | 3 | 1 | 763 | 4.4 |
| rc201 | 10 | optimal | 184.00 | 184.00 | 0.000% | 2 | 1 | 135 | 0.3 |
| rc201 | 25 | optimal | 356.00 | 356.00 | 0.000% | 3 | 1 | 1329 | 24.8 |

Every instance across all three Solomon classes (clustered `c`, random `r`,
mixed `rc`, both narrow "1" and wide "2" time-window variants) is solved to a
*certified* optimum within the 60-second budget. Most root LP relaxations are
already integral (`nodes_explored = 1`); `r201/25` and `r101/25` are the only
cases here that actually require branching, confirming the branch-and-price
layer is exercised and correct, not just idle. A separate run on `r101` with
50 customers also reached a certified optimum (cost 935.00) in 2.0 seconds.
Full raw output: `results/benchmark.csv`.

`gap_percent` is `(cost - lower_bound) / cost`; `0.000%` means the branch-
and-price tree was fully explored and the solution is a certified integer
optimum, not just a heuristic value.

### On comparing against literature "best-known" values

Classic Solomon-benchmark tables report a **lexicographic** objective
(minimize vehicle count first, then distance), whereas this solver's default
objective minimizes distance subject to `sum(y_r) <= K` only. Quoting a
specific "literature-optimal" distance figure without first confirming the
vehicle count matches would be an apples-to-oranges comparison, so no
literature values are hardcoded here. `vrptw_cg/benchmark.py` accepts a
`--reference-csv` with columns `instance,customer_count,best_known_distance`
that you fill in yourself from a citable source (e.g. the SINTEF Solomon
benchmark pages linked in `README.md`) and reports `reference_gap_percent`
alongside the proven `gap_percent` -- keeping the "did we match the
literature" claim auditable rather than asserted.

## 10. Known limitations and future work

* **Performance.** The label-setting pricing algorithm is pure Python; it is
  correct and reasonably fast at Solomon scale (`n <= 50`, see results
  above) but is not competitive with C++ implementations like PyVRP or
  VRPSolver/BaPCod at `n` in the hundreds. This is a deliberate scope choice
  favoring auditability over raw speed.
* **ng-route vs. full elementarity.** With `k=8` neighbors, ng-route pricing
  can in principle miss some negative-reduced-cost elementary routes if the
  cycle it would need to break involves customers outside every relevant
  neighbor set; branch-and-price's arc and Ryan-Foster branching still
  converge to the correct integer optimum regardless (verified in Section
  8), but the root LP bound may be marginally looser than a full bucket-graph
  implementation's.
* **Stabilization center reset per node.** The dual smoothing center
  (Section 6) is currently reset at the start of every branch-and-price
  node rather than warm-started from the parent; this is the simpler,
  unambiguously-correct choice, at the cost of some avoidable pricing calls
  deep in the tree.
* Extensions considered but out of scope here (see
  `GAP_ANALYSIS_AND_ROADMAP.md` for the full list with priorities):
  ML/GNN-guided arc pruning for the pricing graph, Electric-VRPTW, and real
  road-network distance matrices.

## References

See `README.md`.
