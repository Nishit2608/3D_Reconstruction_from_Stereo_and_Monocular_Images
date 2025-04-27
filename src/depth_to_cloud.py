import numpy as np
import open3d as o3d

def depth_to_pointcloud(depth_map, image, K):
    """
    Convert a depth map to a 3D point cloud (with color).
    """
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            z = depth_map[v, u]
            if not np.isnan(z) and z < 50:  # avoid NaNs and far-outliers
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(image[v, u] / 255.0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(points))
    pc.colors = o3d.utility.Vector3dVector(np.array(colors))

    return pc
