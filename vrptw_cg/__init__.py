"""Exact Branch-and-Price solver for the Vehicle Routing Problem with Time Windows.

The package implements:
  * Dantzig-Wolfe / set-partitioning master problem (vrptw_cg.master)
  * ng-route relaxation label-setting pricing algorithm (vrptw_cg.pricing)
  * A branch-and-bound layer on top of column generation, i.e. full
    Branch-and-Price, to close the integrality gap left by plain
    column generation (vrptw_cg.branch_and_price)
  * A constructive heuristic used to seed the root node (vrptw_cg.heuristics)
"""

from .data import Instance, load_instance

__all__ = ["Instance", "load_instance"]
__version__ = "0.2.0"
