import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, cKDTree


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def span(arr, p_low=None, p_high=None):

    if p_low is None:
        return arr.max() - arr.min()

    return np.percentile(arr, p_high) - np.percentile(arr, p_low)


def minmax(arr, p_low=None, p_high=None):

    if p_low is None:
        return arr.min(), arr.max()

    return np.percentile(arr, p_low), np.percentile(arr, p_high)


def convex_area(points_2d):

    if len(points_2d) < 3:
        return 0

    return ConvexHull(points_2d).volume


# ------------------------------------------------------------
# Crown profile
# ------------------------------------------------------------

def crown_profile(points, slices=10):

    z = points[:, 2]

    z0, z1 = minmax(z, 5, 95)

    bins = np.linspace(z0, z1, slices + 1)

    diameters = []

    for i in range(slices):

        mask = (z >= bins[i]) & (z < bins[i + 1])

        pts = points[mask]

        if len(pts) < 50:
            diameters.append(np.nan)
            continue

        xy = pts[:, :2]

        dx = span(xy[:, 0], 5, 95)
        dy = span(xy[:, 1], 5, 95)

        diameters.append(max(dx, dy))

    diameters = np.array(diameters)

    return {

        "mean": np.nanmean(diameters),
        "max": np.nanmax(diameters),
        "p90": np.nanpercentile(diameters, 90),
        "coverage": np.sum(~np.isnan(diameters)),
        "slices": slices
    }


# ------------------------------------------------------------
# Voxel volume
# ------------------------------------------------------------

def voxel_volume(pcd, size):

    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, size)

    n = len(vg.get_voxels())

    return n, n * size ** 3


# ------------------------------------------------------------
# Leaf angles
# ------------------------------------------------------------

def leaf_angles(pcd):

    pcd = pcd.voxel_down_sample(2)

    pcd.estimate_normals()

    normals = np.asarray(pcd.normals)

    cos = np.clip(np.abs(normals[:, 2]), 0, 1)

    angles = np.degrees(np.arccos(cos))

    return {

        "mean": np.mean(angles),
        "median": np.median(angles),
        "p10": np.percentile(angles, 10),
        "p90": np.percentile(angles, 90)
    }


# ------------------------------------------------------------
# Surface area
# ------------------------------------------------------------

def surface_area(pcd):

    pcd.estimate_normals()

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=7
    )

    mesh.remove_degenerate_triangles()

    return mesh.get_surface_area()


# ------------------------------------------------------------
# Roughness
# ------------------------------------------------------------

def roughness(points, k=30, samples=5000):

    tree = cKDTree(points)

    idx = np.random.choice(len(points), min(samples, len(points)))

    residuals = []

    for i in idx:

        _, nn = tree.query(points[i], k)

        neigh = points[nn]

        mu = neigh.mean(axis=0)

        X = neigh - mu

        _, _, vh = np.linalg.svd(X)

        normal = vh[-1]

        dist = abs(np.dot(points[i] - mu, normal))

        residuals.append(dist)

    return np.median(residuals)


# ------------------------------------------------------------
# PCA canopy shape
# ------------------------------------------------------------

def canopy_pca(points):

    xy = points[:, :2]

    xy -= xy.mean(axis=0)

    cov = np.cov(xy.T)

    w, _ = np.linalg.eig(cov)

    w = sorted(w, reverse=True)

    major = 4 * np.sqrt(w[0])
    minor = 4 * np.sqrt(w[1])

    anisotropy = w[0] / w[1]

    return major, minor, anisotropy


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--plant_id", default="002")
    parser.add_argument("--voxel", type=float, default=3)

    args = parser.parse_args()

    path = Path(f"outputs/models/{args.plant_id}/sfm/dense/plant_clean.ply")

    print("[INFO] Loading:", path)

    pcd = o3d.io.read_point_cloud(str(path))

    points = np.asarray(pcd.points)

    print("[INFO] Points:", len(points))

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # ------------------------------------------------------------
    # HEIGHT
    # ------------------------------------------------------------

    zmin, zmax = minmax(z)
    z1, z99 = minmax(z, 1, 99)
    z5, z95 = minmax(z, 5, 95)

    height_raw = zmax - zmin
    height_p1p99 = z99 - z1
    height_p5p95 = z95 - z5

    # ------------------------------------------------------------
    # CANOPY DIAMETER
    # ------------------------------------------------------------

    diam_raw = max(span(x), span(y))
    diam_p1p99 = max(span(x, 1, 99), span(y, 1, 99))
    diam_p5p95 = max(span(x, 5, 95), span(y, 5, 95))

    pca_major, pca_minor, anisotropy = canopy_pca(points)

    slenderness = height_p5p95 / diam_p5p95

    # ------------------------------------------------------------
    # AREAS
    # ------------------------------------------------------------

    area_xy = convex_area(points[:, :2])
    area_xz = convex_area(points[:, [0, 2]])
    area_yz = convex_area(points[:, [1, 2]])

    # ------------------------------------------------------------
    # VOLUMES
    # ------------------------------------------------------------

    hull_volume = ConvexHull(points).volume

    voxels, voxel_vol = voxel_volume(pcd, args.voxel)

    compactness = voxel_vol / hull_volume

    # ------------------------------------------------------------
    # STRUCTURE
    # ------------------------------------------------------------

    crown = crown_profile(points)

    leaf = leaf_angles(pcd)

    # ------------------------------------------------------------
    # SURFACE
    # ------------------------------------------------------------

    print("[INFO] Estimating mesh surface area (~10s)...")

    surface = surface_area(pcd)

    rough = roughness(points)

    # ------------------------------------------------------------
    # QC
    # ------------------------------------------------------------

    bbox = pcd.get_axis_aligned_bounding_box()

    extent = bbox.get_extent()

    bbox_volume = extent[0] * extent[1] * extent[2]

    density = len(points) / bbox_volume

    # ------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------

    print("\n================ HEIGHT =================")

    print(f"Height raw: {height_raw:.2f} mm")
    print(f"Height p1-p99: {height_p1p99:.2f} mm")
    print(f"Height p5-p95: {height_p5p95:.2f} mm")

    print("\n================ CANOPY =================")

    print(f"Diameter raw: {diam_raw:.2f} mm")
    print(f"Diameter p1-p99: {diam_p1p99:.2f} mm")
    print(f"Diameter p5-p95: {diam_p5p95:.2f} mm")

    print(f"PCA major: {pca_major:.2f} mm")
    print(f"PCA minor: {pca_minor:.2f} mm")
    print(f"Anisotropy: {anisotropy:.3f}")

    print(f"Slenderness: {slenderness:.3f}")

    print("\n================ AREAS =================")

    print(f"Projected XY: {area_xy:.2f} mm²")
    print(f"Projected XZ: {area_xz:.2f} mm²")
    print(f"Projected YZ: {area_yz:.2f} mm²")

    print("\n================ VOLUME =================")

    print(f"Convex hull volume: {hull_volume:.2f} mm³")
    print(f"Voxel volume: {voxel_vol:.2f} mm³")
    print(f"Compactness: {compactness:.3f}")

    print("\n================ STRUCTURE =================")

    print(f"Crown mean diameter: {crown['mean']:.2f}")
    print(f"Crown max diameter: {crown['max']:.2f}")
    print(f"Crown p90 diameter: {crown['p90']:.2f}")
    print(f"Crown coverage: {crown['coverage']}/{crown['slices']}")

    print("\n================ LEAF ANGLES =================")

    print(f"Mean: {leaf['mean']:.2f}°")
    print(f"Median: {leaf['median']:.2f}°")
    print(f"P10/P90: {leaf['p10']:.2f} / {leaf['p90']:.2f}")

    print("\n================ SURFACE =================")

    print(f"Surface area proxy: {surface:.2f} mm²")
    print(f"Roughness: {rough:.4f} mm")

    print("\n================ QC =================")

    print(f"Bounding box (mm): {extent}")
    print(f"Point density: {density:.4f} points/mm³")

    print("\n[DONE]")


if __name__ == "__main__":
    main()