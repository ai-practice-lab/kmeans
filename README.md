# K-Means++ 3D demo

K-Means++ clustering implementation with an optional 3D animation of the clustering process for point cloud data.

## Quick start
- Install dependencies: `pip install numpy matplotlib`
- Run with synthetic 3D data (default 4 clusters, saves a GIF):  
  `python main.py`
- Supply your own point cloud (CSV with three columns or `.npy` of shape `(n, 3)`):  
  `python main.py --input path/to/points.csv --clusters 5 --save output/run.gif`
- Disable interactive display if you only want the file:  
  `python main.py --no-show`

## Files
- `kmeans_pp.py` — K-Means++ algorithm with history tracking for visualization.
- `main.py` — CLI to load data, run clustering, and produce a 3D animation (GIF).

## Notes
- Animation uses `matplotlib.animation.PillowWriter`; saving requires Pillow (included with most matplotlib installs).
- Empty clusters are re-seeded with random data points to keep the algorithm running. Adjust `_update_centers` if you want a different strategy.
- K-Means and even K-Means++ are known to be sensitive to initialization; this implementation supports multiple runs with different initializations for better chance of finding the global optimum.
