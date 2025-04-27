# import cv2
# import numpy as np
# import os
# from src.feature_matching import detect_and_match_features
# from src.ransac import ransac_estimator
# from src.lse import least_squares_estimation
# from src.pose import pose_candidates_from_E
# from src.reconstruct_3d import reconstruct3D
# # from src.visualise_pointcloud import visualize_pointcloud           
# from src.plot_epi import plot_epipolar_lines
# from src.dense_depth import compute_dense_depth
# from src.depth_to_cloud import depth_to_pointcloud
# import open3d as o3d
from src.midas_depth import run_midas
# import matplotlib.pyplot as plt
# from src.visualise_pointcloud import visualize_midas_pointcloud

# # import timm

 
 

# def normalize_points(pts, K_inv):
#     """
#     Normalize points using inverse of camera intrinsics.
#     Args:
#         pts: Nx2 array (pixel coordinates)
#         K_inv: 3x3 inverse intrinsic matrix
#     Returns:
#         Nx3 normalized points in homogeneous coordinates
#     """
#     pts_homog = np.hstack((pts, np.ones((pts.shape[0], 1))))  # (N, 3)
#     normalized_pts = (K_inv @ pts_homog.T).T  # (N, 3)
#     return normalized_pts

# def main():
#     # Create results folder if it doesn't exist
#     os.makedirs('results', exist_ok=True)

#     # Load stereo images
#     img1 = cv2.imread('data/images/imagess/000000_10_left.png')
#     img2 = cv2.imread('data/images/imagess/000000_10.png')

#     # Convert to grayscale
#     img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     # Detect and match features
#     kp1, kp2, good_matches, pts1, pts2 = detect_and_match_features(img1_gray, img2_gray)

#     # Draw first 50 matches and save
#     matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=2)
#     cv2.imwrite('results/matched_features.png', matched_img)

#     # KITTI sample camera intrinsics matrix
#     K = np.array([
#         [718.8560, 0, 607.1928],
#         [0, 718.8560, 185.2157],
#         [0, 0, 1]
#     ])
#     K_inv = np.linalg.inv(K)

#     # Normalize matched points
#     pts1_norm = normalize_points(pts1, K_inv)
#     pts2_norm = normalize_points(pts2, K_inv)

#     # Run RANSAC to estimate Essential Matrix
#     E, inlier_indices = ransac_estimator(pts1_norm, pts2_norm)

#     # Select inliers
#     pts1_inliers = pts1_norm[inlier_indices]
#     pts2_inliers = pts2_norm[inlier_indices]

#     print(f"Number of inliers after RANSAC: {len(pts1_inliers)}")

#     # Decompose Essential Matrix to (R, T) candidates
#     pose_candidates = pose_candidates_from_E(E)

#     # Triangulate and select best (R, T)
#     P1, P2, T, R = reconstruct3D(pose_candidates, pts1_inliers, pts2_inliers)

#     # Save final pose
#     np.save('results/final_rotation.npy', R)
#     np.save('results/final_translation.npy', T)

#     print("\nEstimated Rotation Matrix (R):\n", R)
#     print("\nEstimated Translation Vector (T):\n", T)
    
#     # Save final pose
#     np.save('results/final_rotation.npy', R)
#     np.save('results/final_translation.npy', T)

#     print("\nEstimated Rotation Matrix (R):\n", R)
#     print("\nEstimated Translation Vector (T):\n", T)

#     # Visualize 3D points
#     P1_3D = P1  # or (P1 + P2) / 2 for midpoint cloud

#     # Read RGB image before using it
#     rgb_img = cv2.imread(R"C:\Users\nishi\OneDrive\Desktop\Neu\Job applications\Projects\3D_reconstruction_using_lidar\data\images\imagess\000000_10.png")
#     rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

#     visualize_midas_pointcloud(P1_3D, rgb_img)

    
#        # Plot epipolar lines
#     pts1_uncal = np.hstack((pts1[inlier_indices], np.ones((len(inlier_indices), 1)))).T
#     pts2_uncal = np.hstack((pts2[inlier_indices], np.ones((len(inlier_indices), 1)))).T
#     plot_epipolar_lines(img1, img2, pts1_uncal, pts2_uncal, E, K, plot=True)
    
#     # === DENSE DEPTH VISUALIZATION ===
#     print("Computing dense disparity and depth map...")
#     depth_map, disparity = compute_dense_depth(img1_gray, img2_gray, K)

#     # Resize color image to match depth
#     img1_small = cv2.resize(img1, (disparity.shape[1], disparity.shape[0]))

#     # Generate point cloud from depth map
#     print("Converting depth map to point cloud...")
#     dense_pcd = depth_to_pointcloud(depth_map, cv2.cvtColor(img1_small, cv2.COLOR_BGR2RGB), K)
    
#     print(f"Dense point cloud has {len(dense_pcd.points)} points.")

#     # Visualize dense point cloud
#     o3d.visualization.draw_geometries([dense_pcd])
    
#     # Run MiDaS on left image
#     print("Running MiDaS for monocular depth...")
#     depth_map = run_midas(R"C:\Users\nishi\OneDrive\Desktop\Neu\Job applications\Projects\3D_reconstruction_using_lidar\data\images\imagess\000000_10.png")  # <-- change filename if needed

#     # Normalize depth for visualization
#     depth_map_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

#     # Save and show depth map
#     plt.imsave('results/midas_depth.png', depth_map_vis, cmap='plasma')
#     plt.imshow(depth_map_vis, cmap='plasma')
#     plt.title("Monocular Depth Prediction")
#     plt.axis('off')
#     plt.show()
    
#     # # After running MiDaS and getting depth_map
#     # rgb_img = cv2.imread(R"C:\Users\nishi\OneDrive\Desktop\Neu\Job applications\Projects\3D_reconstruction_using_lidar\data\images\imagess\000000_10.png")  # Same image used in MiDaS
#     # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

#     # visualize_midas_pointcloud(depth_map, rgb_img)




# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d

from src.feature_matching import detect_and_match_features
from src.ransac import ransac_estimator
from src.lse import least_squares_estimation
from src.pose import pose_candidates_from_E
from src.reconstruct_3d import reconstruct3D
from src.plot_epi import plot_epipolar_lines
from src.dense_depth import compute_dense_depth
from src.depth_to_cloud import depth_to_pointcloud


def normalize_points(pts, K_inv):
    pts_homog = np.hstack((pts, np.ones((pts.shape[0], 1))))
    normalized_pts = (K_inv @ pts_homog.T).T
    return normalized_pts

def main():
    os.makedirs('results', exist_ok=True)

    # Load stereo images
    img1 = cv2.imread('data/images/imagess/000047_11_left.png')
    img2 = cv2.imread('data/images/imagess/000047_11.png')

    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect and match features
    kp1, kp2, good_matches, pts1, pts2 = detect_and_match_features(img1_gray, img2_gray)
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=2)
    cv2.imwrite('results/matched_features.png', matched_img)

    # Camera intrinsics (KITTI)
    K = np.array([
        [718.8560, 0, 607.1928],
        [0, 718.8560, 185.2157],
        [0, 0, 1]
    ])
    K_inv = np.linalg.inv(K)

    pts1_norm = normalize_points(pts1, K_inv)
    pts2_norm = normalize_points(pts2, K_inv)

    # RANSAC + Essential matrix
    E, inlier_indices = ransac_estimator(pts1_norm, pts2_norm)
    pts1_inliers = pts1_norm[inlier_indices]
    pts2_inliers = pts2_norm[inlier_indices]
    print(f"Number of inliers after RANSAC: {len(pts1_inliers)}")

    # Decompose and select pose
    pose_candidates = pose_candidates_from_E(E)
    P1, P2, T, R = reconstruct3D(pose_candidates, pts1_inliers, pts2_inliers)
    print("\nEstimated Rotation Matrix (R):\n", R)
    print("\nEstimated Translation Vector (T):\n", T)

    np.save('results/final_rotation.npy', R)
    np.save('results/final_translation.npy', T)

    # Epipolar lines
    pts1_uncal = np.hstack((pts1[inlier_indices], np.ones((len(inlier_indices), 1)))).T
    pts2_uncal = np.hstack((pts2[inlier_indices], np.ones((len(inlier_indices), 1)))).T
    plot_epipolar_lines(img1, img2, pts1_uncal, pts2_uncal, E, K, plot=True)

    # === DENSE DISPARITY + DEPTH ===
    print("Computing dense disparity and depth map...")
    depth_map, disparity = compute_dense_depth(img1_gray, img2_gray, K)
    img1_small = cv2.resize(img1, (disparity.shape[1], disparity.shape[0]))
    
    print("Converting depth map to point cloud...")
    dense_pcd = depth_to_pointcloud(depth_map, cv2.cvtColor(img1_small, cv2.COLOR_BGR2RGB), K)
    print(f"Dense point cloud has {len(dense_pcd.points)} points.")
    
    o3d.visualization.draw_geometries([dense_pcd])
    
        # Run MiDaS on left image
    print("Running MiDaS for monocular depth...")
    depth_map = run_midas(
        R"C:\Users\nishi\OneDrive\Desktop\Neu\Job applications\Projects\3D_reconstruction_using_lidar\data\images\imagess\000047_11.png"
    )  # <-- change filename if needed

    # Normalize depth for visualization
    depth_map_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Save and show depth map
    plt.imsave('results/midas_depth.png', depth_map_vis, cmap='plasma')
    plt.imshow(depth_map_vis, cmap='plasma')
    plt.title("Monocular Depth Prediction")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
