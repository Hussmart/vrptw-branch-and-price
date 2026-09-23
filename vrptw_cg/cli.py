"""Command-line entry point.

Replaces the original ``input()``-driven interaction in ``utilities.py``
(``readInstanceN``) with proper CLI arguments, so a run can be scripted,
batched (see benchmark.py) and reproduced exactly from a shell history line.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

from .branch_and_price import BranchAndPrice, solve_lexicographic
from .data import load_instance
from .visualize import plot_convergence, plot_routes

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("vrptw_cg")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vrptw-cg",
        description="Exact Branch-and-Price solver for the VRPTW (Solomon benchmark).")
    p.add_argument("--instance", required=True,
                    help="Solomon instance name, e.g. c101, r101, rc201 (with or without .txt)")
    p.add_argument("--customers", "-n", type=int, required=True,
                    help="number of customers to load from the instance (1-100)")
    p.add_argument("--solomon-dir", default="solomon-instances")
    p.add_argument("--backend", choices=["highs", "gurobi"], default="highs",
                    help="LP solver for the restricted master problem (default: highs, free)")
    p.add_argument("--objective", choices=["distance", "vehicles-then-distance"],
                    default="distance",
                    help="'distance' minimizes distance subject to sum(y_r) <= K (default); "
                         "'vehicles-then-distance' is the standard two-phase Solomon-benchmark "
                         "objective (minimize fleet size first, then distance)")
    p.add_argument("--stabilization-alpha", type=float, default=0.5,
                    help="dual-smoothing weight in [0,1) for column generation, 0 disables it")
    p.add_argument("--time-limit", type=float, default=180.0, help="seconds (per phase)")
    p.add_argument("--node-limit", type=int, default=2000, help="branch-and-price nodes (per phase)")
    p.add_argument("--neighbor-k", type=int, default=8,
                    help="ng-route neighborhood size (larger = closer to exact, slower)")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--plot", action="store_true", help="save a route plot (PNG)")
    p.add_argument("--no-time-window-reduction", action="store_true")
    return p


def run(args=None):
    args = build_parser().parse_args(args)

    inst = load_instance(args.instance, args.customers, solomon_dir=args.solomon_dir,
                          reduce_time_windows=not args.no_time_window_reduction,
                          neighbor_k=args.neighbor_k)
    log.info("Loaded %s with %d customers (K=%d vehicles, Q=%.0f capacity)",
              inst.name, inst.n, inst.K, inst.Q)

    t0 = time.time()
    if args.objective == "vehicles-then-distance":
        lex = solve_lexicographic(inst, backend=args.backend, time_limit=args.time_limit,
                                   node_limit=args.node_limit,
                                   stabilization_alpha=args.stabilization_alpha)
        result = lex.phase2
        vehicles_used = lex.vehicles
        log.info("Phase 1 (minimize vehicles): K* = %d (status %s)",
                  lex.vehicles, lex.phase1.status)
        log.info("Phase 2 (minimize distance with %d vehicles): status %s",
                  lex.vehicles, lex.phase2.status)
    else:
        solver = BranchAndPrice(inst, backend=args.backend, time_limit=args.time_limit,
                                 node_limit=args.node_limit,
                                 stabilization_alpha=args.stabilization_alpha)
        result = solver.solve()
        vehicles_used = len(result.incumbent_routes)
    elapsed = time.time() - t0

    gap = (result.incumbent_cost - result.lower_bound) / result.incumbent_cost * 100 \
        if result.incumbent_cost else 0.0

    log.info("Status: %s", result.status)
    log.info("Best solution cost: %.2f", result.incumbent_cost)
    log.info("Proven lower bound: %.2f (gap %.3f%%)", result.lower_bound, gap)
    log.info("Vehicles used: %d", vehicles_used)
    log.info("Nodes explored: %d (pruned by bound: %d, infeasible: %d)",
              result.stats.nodes_explored, result.stats.nodes_pruned_bound,
              result.stats.nodes_pruned_infeasible)
    log.info("Columns generated: %d", result.stats.columns_generated)
    log.info("Time: %.1fs (limit %.0fs per phase)", elapsed, args.time_limit)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                             f"{inst.name}-{inst.n}customers.json")
    payload = {
        "instance": inst.name,
        "customers": inst.n,
        "backend": args.backend,
        "objective": args.objective,
        "status": result.status,
        "cost": result.incumbent_cost,
        "lower_bound": result.lower_bound,
        "gap_percent": gap,
        "vehicles_used": vehicles_used,
        "routes": result.incumbent_routes,
        "nodes_explored": result.stats.nodes_explored,
        "columns_generated": result.stats.columns_generated,
        "elapsed_seconds": elapsed,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Results written to %s", out_path)

    if args.plot:
        plot_path = os.path.join(args.output_dir, f"{inst.name}-{inst.n}customers-routes.png")
        plot_routes(inst, result.incumbent_routes,
                    title=f"{inst.name} n={inst.n}: cost={result.incumbent_cost:.1f}, "
                          f"{vehicles_used} vehicles",
                    save_path=plot_path)
        log.info("Route plot written to %s", plot_path)

        if result.stats.root_iteration_bounds:
            conv_path = os.path.join(args.output_dir,
                                      f"{inst.name}-{inst.n}customers-convergence.png")
            plot_convergence(result.stats.root_iteration_bounds,
                              title=f"{inst.name} n={inst.n}: root column generation",
                              save_path=conv_path)
            log.info("Convergence plot written to %s", conv_path)

    return result


if __name__ == "__main__":
    run()
