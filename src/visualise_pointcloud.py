import numpy as np
import open3d as o3d
import cv2

def visualize_midas_pointcloud(depth_map, rgb_img, fx=500.0, fy=500.0, cx=None, cy=None):
    h, w = depth_map.shape
    if cx is None: cx = w / 2.0
    if cy is None: cy = h / 2.0

    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth_map
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    # Stack 3D points
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Normalize and reshape RGB image
    rgb_img = cv2.resize(rgb_img, (w, h))
    colors = rgb_img.reshape(-1, 3) / 255.0

    # Remove invalid depth values
    mask = (z > 0).reshape(-1)
    points = points[mask]
    colors = colors[mask]

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"[INFO] Visualizing MiDaS point cloud with {len(points)} points...")
    o3d.visualization.draw_geometries([pcd])
