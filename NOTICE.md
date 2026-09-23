# Provenance

This repository is a fork of
[`SimoneRichetti/VRPTW-Column-Generation`](https://github.com/SimoneRichetti/VRPTW-Column-Generation)
by Simone Richetti, which implemented single-pass Column Generation (LP
relaxation only) for the VRPTW using Gurobi, based on Desrochers, Desrosiers
& Solomon (1992) and the IMPACT construction heuristic of Ioannou, Kritikos
& Prastacos (2001).

* The original code is kept unmodified in [`legacy/`](legacy/) for reference
  and side-by-side benchmarking.
* [`GAP_ANALYSIS_AND_ROADMAP.md`](GAP_ANALYSIS_AND_ROADMAP.md) documents,
  file by file, what the original implementation did and did not do.
* [`vrptw_cg/`](vrptw_cg/) is a new implementation that reuses the same
  Solomon data format and the IMPACT heuristic idea, but adds a full
  Branch-and-Price layer (the original never branched -- see the gap
  analysis), ng-route relaxation pricing, a free/open-source LP backend, and
  test coverage validated against an independent brute-force oracle. See
  [`docs/REPORT.md`](docs/REPORT.md) for the technical writeup.

No code from `legacy/` is imported or executed by `vrptw_cg/`; it is kept
purely as a reference point.
