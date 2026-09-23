# VRPTW Branch-and-Price

An exact solver for the **Vehicle Routing Problem with Time Windows (VRPTW)**
built around **Branch-and-Price**: Dantzig-Wolfe column generation with an
**ng-route relaxation** pricing algorithm, wrapped in a branch-and-bound
layer that closes the integrality gap and *proves* optimality (not just a
good heuristic number).

This started as a fork of
[`SimoneRichetti/VRPTW-Column-Generation`](https://github.com/SimoneRichetti/VRPTW-Column-Generation),
which solved only the LP relaxation of the set-partitioning master problem.
The original code is preserved under [`legacy/`](legacy/); everything under
[`vrptw_cg/`](vrptw_cg/) is a new implementation. See
[`GAP_ANALYSIS_AND_ROADMAP.md`](GAP_ANALYSIS_AND_ROADMAP.md) for a detailed,
file-by-file account of what changed and why, and
[`docs/REPORT.md`](docs/REPORT.md) for the full technical writeup
(formulation, algorithms, complexity, and validation).

## What "exact" means here

Column generation alone only gives a **lower bound** and a generally
fractional LP solution. This project adds the missing branch-and-bound
layer on top (Barnhart et al., 1998), so the number it reports is a
certified integer optimum -- or, if a time/node budget is hit, a feasible
solution together with a proven lower bound and optimality gap.

Correctness is checked two ways, not just asserted:

1. **Dual-feasibility tests** (`tests/test_master.py`) verify the LP
   optimality condition (every column's reduced cost `>= 0` at the optimum)
   against SciPy's HiGHS backend directly -- this is what catches a sign
   error in the reduced-cost formula before it silently breaks column
   generation.
2. **An independent bitmask-DP brute-force solver**
   (`tests/test_branch_and_price.py`), unrelated to column generation, is
   used as ground truth on small instances. Branch-and-Price matches it
   exactly on every tested case.

## Quickstart

```bash
git clone https://github.com/Hussmart/vrptw-branch-and-price.git
cd vrptw-branch-and-price
pip install -r requirements.txt          # numpy, scipy, matplotlib, pytest -- no license needed
python -m vrptw_cg.cli --instance c101 --customers 25 --plot
```

```
Loaded c101 with 25 customers (K=25 vehicles, Q=200 capacity)
Status: optimal
Best solution cost: 192.00
Proven lower bound: 192.00 (gap 0.000%)
Vehicles used: 3
Time: 26.8s (limit 180s)
```

Batch benchmark across the Solomon suite:

```bash
python -m vrptw_cg.benchmark --instances c101 r101 rc101 --customer-counts 10 25
```

Run the test suite:

```bash
pytest
```

## Why this is different from the original project

| | `legacy/` (original) | `vrptw_cg/` (this fork) |
|---|---|---|
| Integrality | LP relaxation only, rounded with a greedy heuristic | Full Branch-and-Price -- certified integer optimum |
| Pricing subproblem | Non-elementary resource DP; can revisit a customer | ng-route relaxation label-setting (Baldacci et al., 2011) |
| Fleet-size constraint | Accepted as a parameter but never enforced | Enforced, and branched on (Desrochers-Desrosiers-Solomon, 1992) |
| Solver | Gurobi only (license required) | Free HiGHS backend by default; Gurobi optional |
| Interface | Interactive `input()`, one instance at a time | CLI + batch benchmark script |
| Visualization | None (a TODO in the original code) | Route maps and convergence plots |
| Tests | None | Unit tests + independent brute-force validation |

## Solver architecture

```
vrptw_cg/
  data.py              instance loading, distance matrix, time-window reduction, ng-route neighbor sets
  master.py            set-partitioning LP relaxation (SciPy/HiGHS, optional Gurobi), with artificial
                        variables so every branch-and-price node is always LP-feasible
  pricing.py           ng-route relaxation label-setting algorithm (the column generation subproblem)
  branch_and_price.py  the branch-and-bound layer: vehicle-count branching, arc branching,
                        and Ryan-Foster branching as the provably complete fallback
  heuristics.py        IMPACT construction heuristic, used only to seed the root node
  visualize.py         route maps and column-generation convergence plots
  cli.py               command-line entry point
  benchmark.py         batch runner across the Solomon benchmark
```

## On using ORTEC

ORTEC is a commercial logistics-optimization vendor; its production solver
is closed-source with no public API, so it cannot be embedded here directly.
What *is* real and used as a reference point in `docs/REPORT.md`: ORTEC
maintains an [open-source fork of PyVRP](https://github.com/ortec/PyVRP)
(MIT license, first place in the DIMACS VRPTW 2021 and EURO meets NeurIPS
2022 competitions) and has organized several real-data academic competitions
(the VeRoLog Solver Challenges, and EURO meets NeurIPS 2022). Those are the
legitimate, citable way to benchmark this solver against both industrial
practice and the current open-source state of the art.

## References

* M. Desrochers, J. Desrosiers, M. Solomon, "A New Optimization Algorithm for
  the Vehicle Routing Problem with Time Windows", *Operations Research*, 1992.
* C. Barnhart et al., "Branch-and-Price: Column Generation for Solving Huge
  Integer Programs", *Operations Research*, 1998.
* R. Baldacci, A. Mingozzi, R. Roberti, "New route relaxation and pricing
  strategies for the vehicle routing problem", *Operations Research*, 2011.
* D. Ryan, B. Foster, "An integer programming approach to scheduling", 1981.
* R. Sadykov, E. Uchoa, A. Pessoa, "A Bucket Graph Based Labeling Algorithm
  with Application to Vehicle Routing", *Transportation Science*, 2021.
* W. Kool et al. / PyVRP contributors, "PyVRP: A High-Performance VRP Solver
  Package", *INFORMS Journal on Computing*, 2024.
* G. Ioannou, M. Kritikos, G. Prastacos, "A greedy look-ahead heuristic for
  the vehicle routing problem", *JORS*, 2001 (the IMPACT heuristic used to
  seed the root node).
* M. Solomon, "VRPTW Benchmark Problems", http://web.cba.neu.edu/~msolomon/problems.htm

See `NOTICE.md` for the fork's provenance and `LICENSE` for terms.
