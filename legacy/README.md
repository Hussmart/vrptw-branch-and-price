# Original implementation (kept for reference)

This folder holds the original Column Generation implementation by
Simone Richetti (upstream: `SimoneRichetti/VRPTW-Column-Generation`), kept
unmodified so the improvements in `vrptw_cg/` can be read and benchmarked
against it directly.

It is **not used** by the new solver. In particular:

* `col-gen-vrptw.py` only solves the LP relaxation of the set-partitioning
  master problem -- there is no branch-and-bound layer, so the reported
  solution is not guaranteed to be integer-optimal.
* `ESPmodel.py` (the exact elementary-shortest-path pricing MIP) times out
  after ~160 minutes even on small instances -- see `note.txt` -- and is not
  actually usable.
* `optimization.py`'s `subProblem` solves a *non-elementary* resource-indexed
  shortest path (Desrochers et al., 1992), which can return routes that
  revisit a customer.
* The master problem never enforces the fleet-size constraint
  (`vehicleNumber` is accepted but unused in `createMasterProblem`).

See `../GAP_ANALYSIS_AND_ROADMAP.md` for the full analysis and
`../docs/REPORT.md` for how `vrptw_cg/` addresses each point.
