"""Example runner for 3D K-Means++ with clustering animation."""

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from kmeans_pp import KMeansPP


def load_points(path: Optional[str], seed: int, samples: int) -> np.ndarray:
    """Load Nx3 points from disk or generate a synthetic cloud."""
    if path:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if path_obj.suffix.lower() == ".npy":
            points = np.load(path_obj)
        else:
            points = np.loadtxt(path_obj, delimiter=",")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected an array with shape (n_samples, 3).")
        return points.astype(float)

    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            (-1.5, -1.0, -0.8),
            (0.8, 0.4, 1.2),
            (1.6, -1.4, -0.4),
            (-0.2, 1.6, 0.6),
        ]
    )
    points = []
    per_cluster = max(1, samples // len(centers))
    for c in centers:
        points.append(rng.normal(loc=c, scale=0.35, size=(per_cluster, 3)))
    return np.vstack(points)


def animate_kmeans_3d(
    X: np.ndarray,
    centers_history: Sequence[np.ndarray],
    labels_history: Sequence[np.ndarray],
    interval: int = 700,
    save_path: Optional[str] = None,
    show: bool = True,
    all_centers_history: Optional[Sequence[Sequence[np.ndarray]]] = None,
    all_labels_history: Optional[Sequence[Sequence[np.ndarray]]] = None,
    best_run_index: Optional[int] = None,
    best_inertia: Optional[float] = None,
):
    """Animate clustering progress in 3D using history recorded during fit."""
    if all_centers_history is not None and all_labels_history is not None:
        runs_centers = list(all_centers_history)
        runs_labels = list(all_labels_history)
        if len(runs_centers) != len(runs_labels):
            raise ValueError("all_centers_history and all_labels_history length mismatch.")
    else:
        runs_centers = [centers_history]
        runs_labels = [labels_history]

    if any(len(lh) == 0 or len(ch) == 0 for lh, ch in zip(runs_labels, runs_centers)):
        raise ValueError("Each run must contain at least one centers_history and labels_history entry.")

    frame_plan = []
    for run_idx, (ch, lh) in enumerate(zip(runs_centers, runs_labels)):
        frames = min(len(lh), len(ch))
        for iter_idx in range(frames):
            frame_plan.append((run_idx, iter_idx))

    if not frame_plan:
        raise ValueError("No frames to animate.")

    palette = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    span = np.ptp(X, axis=0)
    mins = X.min(axis=0) - 0.2 * span
    maxs = X.max(axis=0) + 0.2 * span
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    scatter = ax.scatter([], [], [], s=18, alpha=0.8)
    centers_plot = ax.scatter([], [], [], s=120, marker="X", edgecolors="black", linewidths=1.0)
    subtitle = ax.text2D(0.02, 0.94, "", transform=ax.transAxes)

    def _colors(labels: Iterable[int]) -> np.ndarray:
        labels = np.asarray(labels)
        return palette(labels % 10)

    def init():
        scatter._offsets3d = ([], [], [])
        centers_plot._offsets3d = ([], [], [])
        subtitle.set_text("")
        return scatter, centers_plot, subtitle

    def update(frame: int):
        run_idx, iter_idx = frame_plan[frame]
        labels = runs_labels[run_idx][iter_idx]
        centers = runs_centers[run_idx][iter_idx]
        scatter._offsets3d = (X[:, 0], X[:, 1], X[:, 2])
        scatter.set_color(_colors(labels))

        centers_plot._offsets3d = (centers[:, 0], centers[:, 1], centers[:, 2])
        centers_plot.set_color(palette(np.arange(len(centers)) % 10))
        run_total_frames = min(len(runs_labels[run_idx]), len(runs_centers[run_idx]))
        ax.set_title(f"K-Means++ run {run_idx + 1}/{len(runs_labels)} | iteration {iter_idx + 1}/{run_total_frames}")

        best_suffix = ""
        if best_run_index is not None and best_inertia is not None:
            best_suffix = f" | best run: {best_run_index + 1}, inertia: {best_inertia:.4f}"
        subtitle.set_text(f"Frames: {frame + 1}/{len(frame_plan)}{best_suffix}")
        return scatter, centers_plot, subtitle

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_plan), init_func=init, interval=interval, blit=False, repeat=False
    )

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.PillowWriter(fps=max(1, int(1000 / max(interval, 1))))
        ani.save(out_path, writer=writer)
        print(f"Animation saved to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-Means++ on 3D point clouds with animation.")
    parser.add_argument("--input", type=str, help="Path to .npy or .csv file containing Nx3 points.")
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters.")
    parser.add_argument("--max-iter", type=int, default=25, help="Maximum K-Means iterations.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance on centroid shift.")
    parser.add_argument("--samples", type=int, default=800, help="Synthetic sample size if no input is provided.")
    parser.add_argument("--interval", type=int, default=700, help="Frame interval in ms for animation.")
    parser.add_argument("--n-init", type=int, default=3, help="How many independent K-Means++ initializations to try.")
    parser.add_argument("--save", type=str, default="output/kmeans_animation.gif", help="Path to save GIF animation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive display of the animation.")
    return parser.parse_args()


def main():
    args = parse_args()
    points = load_points(args.input, seed=args.seed, samples=args.samples)

    model = KMeansPP(
        n_clusters=args.clusters,
        max_iter=args.max_iter,
        n_init=args.n_init,
        tol=args.tol,
        random_state=args.seed,
    )
    model.fit(points)

    print(f"Tried {args.n_init} run(s). Best run: {model.best_run_index_ + 1 if model.best_run_index_ is not None else 'N/A'}")
    for idx, inertia in enumerate(model.runs_inertia_, start=1):
        print(f"  Run {idx}: inertia = {inertia:.4f}")
    print(f"Selected inertia: {model.inertia_:.4f}")
    print(f"Centers shape: {model.centers_.shape}, iterations recorded (best run): {len(model.labels_history_)}")

    animate_kmeans_3d(
        points,
        centers_history=model.centers_history_,
        labels_history=model.labels_history_,
        interval=args.interval,
        save_path=args.save,
        show=not args.no_show,
        all_centers_history=model.all_centers_history_,
        all_labels_history=model.all_labels_history_,
        best_run_index=model.best_run_index_,
        best_inertia=model.inertia_,
    )


if __name__ == "__main__":
    main()
