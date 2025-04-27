import cv2
import numpy as np

def compute_dense_depth(imgL, imgR, K, baseline=0.54):
    """
    Compute dense depth map from stereo images using OpenCV's StereoSGBM.
    Args:
        imgL: Left image (grayscale)
        imgR: Right image (grayscale)
        K: Camera intrinsics matrix
        baseline: Distance between stereo cameras (in meters)
    Returns:
        depth_map: Depth values (same shape as input)
    """
    # StereoSGBM Parameters
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # must be divisible by 16
        blockSize=9,
        P1=8 * 3 * 9**2,
        P2=32 * 3 * 9**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan  # avoid invalid depths

    # Convert disparity to depth
    focal_length = K[0, 0]
    depth_map = (focal_length * baseline) / (disparity + 1e-6)

    return depth_map, disparity
