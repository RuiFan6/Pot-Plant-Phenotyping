import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def cluster_diameter(points):

    mn = points.min(axis=0)
    mx = points.max(axis=0)

    return np.max(mx - mn)


def pca_axes(points):

    X = points - points.mean(axis=0)

    cov = np.cov(X.T)

    w, _ = np.linalg.eig(cov)

    w = sorted(w, reverse=True)

    return 4*np.sqrt(w[0]), 4*np.sqrt(w[1]), 4*np.sqrt(w[2])


def voxel_volume(points, voxel):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel)

    n = len(vg.get_voxels())

    return n * voxel**3


def ripeness(colors):

    R = colors[:,0]
    G = colors[:,1]

    idx = R/(R+G+1e-6)

    return np.mean(idx)


# ------------------------------------------------------------
# Split merged clusters
# ------------------------------------------------------------

def split_cluster(points, colors, diameter):

    if diameter < 40:
        return [(points, colors)]

    # estimate fruit count
    n = int(round(diameter/30))

    n = max(2, n)

    print(f"[INFO] Splitting merged cluster into {n} fruits")

    kmeans = KMeans(n_clusters=n, n_init=10)

    labels = kmeans.fit_predict(points)

    clusters = []

    for i in range(n):

        mask = labels == i

        clusters.append((points[mask], colors[mask]))

    return clusters


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--plant_id", default="002")
    parser.add_argument("--eps", type=float, default=10)
    parser.add_argument("--min_points", type=int, default=1500)

    args = parser.parse_args()

    fruit_path = Path(
        f"outputs/models/{args.plant_id}/sfm/dense/fruit_detected.ply"
    )

    print("[INFO] Loading:", fruit_path)

    pcd = o3d.io.read_point_cloud(str(fruit_path))

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    print("[INFO] Points:", len(pts))

    # ------------------------------------------------------------
    # Initial clustering
    # ------------------------------------------------------------

    labels = np.array(
        pcd.cluster_dbscan(
            eps=args.eps,
            min_points=args.min_points
        )
    )

    clusters = labels.max() + 1

    print("[INFO] Initial clusters:", clusters)

    fruit_clusters = []

    for i in range(clusters):

        mask = labels == i

        p = pts[mask]
        c = cols[mask]

        d = cluster_diameter(p)

        split = split_cluster(p, c, d)

        fruit_clusters.extend(split)

    print("[INFO] Final fruit count:", len(fruit_clusters))

    # ------------------------------------------------------------
    # Traits
    # ------------------------------------------------------------

    total_volume = 0

    print("\n================ FRUIT TRAITS ================")

    for i, (p, c) in enumerate(fruit_clusters):

        diam = cluster_diameter(p)

        axes = pca_axes(p)

        vol = voxel_volume(p, 1.5)

        rip = ripeness(c)

        centroid = p.mean(axis=0)

        total_volume += vol

        print(f"\nFruit {i+1}")

        print(f"points: {len(p)}")

        print(f"diameter: {diam:.2f} mm")

        print(f"PCA axes: {axes[0]:.2f}, {axes[1]:.2f}, {axes[2]:.2f} mm")

        print(f"volume: {vol:.2f} mm³")

        print(f"height: {centroid[2]:.2f} mm")

        print(f"ripeness: {rip:.2f}")

    print("\n--------------- SUMMARY ---------------")

    print("Fruit count:", len(fruit_clusters))

    print(f"Total fruit volume: {total_volume:.2f} mm³")

    # ------------------------------------------------------------
    # Save colored clusters
    # ------------------------------------------------------------

    vis_pts = []
    vis_cols = []

    rng = np.random.default_rng(0)

    for p, _ in fruit_clusters:

        col = rng.random(3)

        vis_pts.append(p)

        vis_cols.append(np.tile(col,(len(p),1)))

    vis_pts = np.vstack(vis_pts)
    vis_cols = np.vstack(vis_cols)

    vis = o3d.geometry.PointCloud()

    vis.points = o3d.utility.Vector3dVector(vis_pts)
    vis.colors = o3d.utility.Vector3dVector(vis_cols)

    out = fruit_path.with_name("fruit_clusters_colored.ply")

    o3d.io.write_point_cloud(str(out), vis)

    print("\n[INFO] Saved colored clusters:", out)

    print("\n[DONE]")


if __name__ == "__main__":
    main()