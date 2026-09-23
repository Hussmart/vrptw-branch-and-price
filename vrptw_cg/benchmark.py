"""Batch runner over the Solomon benchmark, producing a CSV suitable for the
technical report (docs/REPORT.md).

Deliberately does NOT hardcode "best known" literature values from memory:
wrong numbers quoted as literature fact would be worse than no comparison at
all, especially for a document meant to be read by an admissions committee.
Instead, ``--reference-csv`` accepts a small CSV (customer_count,instance,
best_known_distance) that the user fills in from a verifiable, citable
source (e.g. the SINTEF Solomon benchmark pages referenced in README.md) --
see docs/REPORT.md for the exact procedure and citation.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time

from .branch_and_price import BranchAndPrice
from .data import load_instance

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("vrptw_cg.benchmark")


def load_reference(path: str | None):
    if not path:
        return {}
    ref = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["instance"].strip().lower(), int(row["customer_count"]))
            ref[key] = float(row["best_known_distance"])
    return ref


def run_benchmark(instances, customer_counts, backend="highs", time_limit=120.0,
                   solomon_dir="solomon-instances", reference=None, out_csv="results/benchmark.csv"):
    reference = reference or {}
    rows = []
    for name in instances:
        for n in customer_counts:
            try:
                inst = load_instance(name, n, solomon_dir=solomon_dir)
            except (FileNotFoundError, ValueError) as exc:
                log.warning("Skipping %s/%d: %s", name, n, exc)
                continue

            solver = BranchAndPrice(inst, backend=backend, time_limit=time_limit)
            t0 = time.time()
            result = solver.solve()
            elapsed = time.time() - t0
            gap = ((result.incumbent_cost - result.lower_bound) / result.incumbent_cost * 100
                   if result.incumbent_cost else 0.0)
            ref_val = reference.get((name.lower(), n))
            ref_gap = ((result.incumbent_cost - ref_val) / ref_val * 100) if ref_val else None

            row = {
                "instance": name, "customers": n, "status": result.status,
                "cost": round(result.incumbent_cost, 2),
                "lower_bound": round(result.lower_bound, 2),
                "gap_percent": round(gap, 3),
                "vehicles": len(result.incumbent_routes),
                "nodes_explored": result.stats.nodes_explored,
                "columns_generated": result.stats.columns_generated,
                "elapsed_seconds": round(elapsed, 1),
                "reference_best_known": ref_val,
                "reference_gap_percent": round(ref_gap, 3) if ref_gap is not None else None,
            }
            rows.append(row)
            log.info("%-8s n=%-3d cost=%-8.2f gap=%-6.2f%% vehicles=%-2d time=%.1fs",
                      name, n, row["cost"], row["gap_percent"], row["vehicles"], elapsed)

    if rows:
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info("Benchmark written to %s (%d rows)", out_csv, len(rows))
    return rows


def build_parser():
    p = argparse.ArgumentParser(description="Batch-run the B&P solver over Solomon instances.")
    p.add_argument("--instances", nargs="+",
                    default=["c101", "c201", "r101", "r201", "rc101", "rc201"])
    p.add_argument("--customer-counts", nargs="+", type=int, default=[10, 25])
    p.add_argument("--backend", choices=["highs", "gurobi"], default="highs")
    p.add_argument("--time-limit", type=float, default=120.0)
    p.add_argument("--solomon-dir", default="solomon-instances")
    p.add_argument("--reference-csv", default=None,
                    help="optional CSV with columns instance,customer_count,best_known_distance")
    p.add_argument("--out-csv", default="results/benchmark.csv")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    reference = load_reference(args.reference_csv)
    run_benchmark(args.instances, args.customer_counts, backend=args.backend,
                  time_limit=args.time_limit, solomon_dir=args.solomon_dir,
                  reference=reference, out_csv=args.out_csv)


if __name__ == "__main__":
    main()
