import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ------------------------------------------------------------
# Color helpers
# ------------------------------------------------------------

def redness(colors):
    R = colors[:,0]
    G = colors[:,1]
    B = colors[:,2]

    return R / (G + B + 1e-6)


# ------------------------------------------------------------
# Seed detection
# ------------------------------------------------------------

def fruit_seed_mask(colors):

    r = redness(colors)

    return r > 1.6


# ------------------------------------------------------------
# Region growing
# ------------------------------------------------------------

def grow_region(seed_points, all_points, colors, radius):

    tree = cKDTree(all_points)

    r_index = redness(colors)

    mask = np.zeros(len(all_points), dtype=bool)

    for p in seed_points:

        idx = tree.query_ball_point(p, radius)

        idx = np.array(idx)

        grow = r_index[idx] > 1.1

        mask[idx[grow]] = True

    return mask


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--plant_id", default="002")

    parser.add_argument("--seed_eps", type=float, default=6,
                        help="DBSCAN radius for seeds")

    parser.add_argument("--grow_radius", type=float, default=18,
                        help="fruit region growing radius")

    parser.add_argument("--min_cluster", type=int, default=2000)

    args = parser.parse_args()

    input_path = Path(
        f"outputs/models/{args.plant_id}/sfm/dense/plant_clean.ply"
    )

    output_path = Path(
        f"outputs/models/{args.plant_id}/sfm/dense/fruit_detected.ply"
    )

    print("[INFO] Loading:", input_path)

    pcd = o3d.io.read_point_cloud(str(input_path))

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    print("[INFO] Points:", len(points))

    # ------------------------------------------------------------
    # Seed detection
    # ------------------------------------------------------------

    seed_mask = fruit_seed_mask(colors)

    seed_indices = np.where(seed_mask)[0]

    seed_pcd = pcd.select_by_index(seed_indices)

    print("[INFO] Seed points:", len(seed_indices))

    # ------------------------------------------------------------
    # Cluster seeds
    # ------------------------------------------------------------

    print("[INFO] Clustering seeds...")

    labels = np.array(
        seed_pcd.cluster_dbscan(
            eps=args.seed_eps,
            min_points=50
        )
    )

    clusters = labels.max() + 1

    print("[INFO] Seed clusters:", clusters)

    seed_points = np.asarray(seed_pcd.points)

    # ------------------------------------------------------------
    # Region growing
    # ------------------------------------------------------------

    print("[INFO] Growing fruit regions...")

    grow_mask = grow_region(
        seed_points,
        points,
        colors,
        args.grow_radius
    )

    fruit_indices = np.where(grow_mask)[0]

    fruit_pcd = pcd.select_by_index(fruit_indices)

    # ------------------------------------------------------------
    # Final clustering
    # ------------------------------------------------------------

    print("[INFO] Final clustering...")

    labels = np.array(
        fruit_pcd.cluster_dbscan(
            eps=10,
            min_points=args.min_cluster
        )
    )

    valid = labels >= 0

    clusters = labels[valid].max() + 1 if valid.any() else 0

    print("[INFO] Fruit clusters detected:", clusters)

    final_indices = np.where(valid)[0]

    fruit_final = fruit_pcd.select_by_index(final_indices)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    o3d.io.write_point_cloud(
        str(output_path),
        fruit_final
    )

    print("[INFO] Saved:", output_path)

    print("\n[DONE]")


if __name__ == "__main__":
    main()