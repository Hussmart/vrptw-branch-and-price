"""Route and convergence plotting.

The original project's own TODO list asked for this ("Plot routes") and
never implemented it. Matplotlib is used (not folium/plotly) so a plot can
be produced headlessly in CI and in the benchmark report without a browser
or an internet map tile server.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_routes(instance, routes, title: str = "", save_path: str | None = None,
                 show: bool = False):
    fig, ax = plt.subplots(figsize=(8, 8))
    x, y = instance.x, instance.y

    ax.scatter(x[1:-1], y[1:-1], c="#4c72b0", s=40, zorder=3, label="customers")
    ax.scatter([x[0]], [y[0]], c="black", marker="s", s=120, zorder=4, label="depot")

    colors = matplotlib.colormaps["tab20"].resampled(max(len(routes), 1))
    for idx, route in enumerate(routes):
        xs = [x[node] for node in route]
        ys = [y[node] for node in route]
        ax.plot(xs, ys, color=colors(idx % 20), linewidth=1.6, zorder=2,
                label=f"route {idx + 1} ({len(route) - 2} stops)")

    ax.set_title(title or f"{instance.name} -- {instance.n} customers, "
                           f"{len(routes)} routes")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if len(routes) <= 15:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_convergence(bounds: list[float], title: str = "", save_path: str | None = None):
    """LP-bound-per-iteration chart for the root node's column generation
    loop, useful to show/discuss the classic "tailing-off" effect."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(bounds) + 1), bounds, marker="o", markersize=3)
    ax.set_xlabel("column generation iteration")
    ax.set_ylabel("root LP objective")
    ax.set_title(title or "Column generation convergence")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
