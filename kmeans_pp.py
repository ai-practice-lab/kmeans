"""Minimal K-Means++ implementation with history tracking."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class KMeansPP:
    """K-Means clustering with K-Means++ initialization."""

    n_clusters: int
    max_iter: int = 100
    n_init: int = 1
    tol: float = 1e-4
    random_state: Optional[int] = None

    centers_: np.ndarray = field(init=False, default=None)
    labels_: np.ndarray = field(init=False, default=None)
    inertia_: float = field(init=False, default=None)
    centers_history_: List[np.ndarray] = field(init=False, default_factory=list)
    labels_history_: List[np.ndarray] = field(init=False, default_factory=list)
    runs_inertia_: List[float] = field(init=False, default_factory=list)
    best_run_index_: Optional[int] = field(init=False, default=None)
    all_centers_history_: List[List[np.ndarray]] = field(init=False, default_factory=list)
    all_labels_history_: List[List[np.ndarray]] = field(init=False, default_factory=list)

    def fit(self, X: np.ndarray) -> "KMeansPP":
        """Run clustering on the given data."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X should be a 2D array of shape (n_samples, n_features).")
        if self.n_clusters <= 0 or self.n_clusters > len(X):
            raise ValueError("n_clusters must be in [1, n_samples].")
        if self.n_init <= 0:
            raise ValueError("n_init must be a positive integer.")

        self.runs_inertia_.clear()
        self.all_centers_history_.clear()
        self.all_labels_history_.clear()

        best_inertia = np.inf
        best_run = None

        for run_idx in range(self.n_init):
            # Offset seed to create distinct initializations while retaining reproducibility.
            seed = None if self.random_state is None else self.random_state + run_idx
            rng = np.random.default_rng(seed)

            centers = self._init_centers(X, rng)
            centers_history = [centers.copy()]
            labels_history = []

            for _ in range(self.max_iter):
                labels = self._assign_labels(X, centers)
                labels_history.append(labels)

                new_centers = self._update_centers(X, labels, rng)
                shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
                centers = new_centers
                centers_history.append(centers.copy())

                if shift <= self.tol:
                    break

            final_labels = self._assign_labels(X, centers)
            inertia = float(np.sum((X - centers[final_labels]) ** 2))

            if not labels_history or not np.array_equal(final_labels, labels_history[-1]):
                labels_history.append(final_labels)
                centers_history.append(centers.copy())

            self.all_centers_history_.append(centers_history)
            self.all_labels_history_.append(labels_history)
            self.runs_inertia_.append(inertia)

            if inertia < best_inertia:
                best_inertia = inertia
                best_run = (
                    centers,
                    final_labels,
                    centers_history,
                    labels_history,
                    run_idx,
                )

        if best_run is None:
            raise RuntimeError("KMeansPP fit failed to complete any initialization.")

        self.centers_, self.labels_, self.centers_history_, self.labels_history_, self.best_run_index_ = (
            best_run[0],
            best_run[1],
            best_run[2],
            best_run[3],
            best_run[4],
        )
        self.inertia_ = best_inertia

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign clusters for new points."""
        if self.centers_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        return self._assign_labels(X, self.centers_)

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """K-Means++ seeding."""
        n_samples = len(X)
        first_idx = rng.integers(0, n_samples)
        centers = [X[first_idx]]
        closest_dist_sq = np.full(n_samples, np.inf)

        for _ in range(1, self.n_clusters):
            new_dist_sq = np.sum((X - centers[-1]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

            probs = closest_dist_sq / closest_dist_sq.sum()
            cumulative = np.cumsum(probs)
            r = rng.random()
            next_idx = np.searchsorted(cumulative, r)
            centers.append(X[next_idx])

        return np.vstack(centers)

    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _update_centers(
        self, X: np.ndarray, labels: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Compute mean per cluster; re-seed empty clusters to random points."""
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                centers[k] = X[rng.integers(0, len(X))]
            else:
                centers[k] = members.mean(axis=0)
        return centers
