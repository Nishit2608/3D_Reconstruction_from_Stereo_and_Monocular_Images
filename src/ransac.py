from src.lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    """
    Performs RANSAC to robustly estimate the Essential Matrix E.
    
    Args:
        X1: N x 3 array of points from image 1 (homogeneous)
        X2: N x 3 array of points from image 2 (homogeneous)
        num_iterations: Number of RANSAC iterations
    
    Returns:
        best_E: Best Essential Matrix found
        best_inliers: Indices of inlier matches
    """
    sample_size = 8
    eps = 1e-4
    best_num_inliers = -1
    best_E = None
    best_inliers = None

    for i in range(num_iterations):
        perm = np.random.RandomState(seed=i).permutation(len(X1))
        sample_idx, test_idx = perm[:sample_size], perm[sample_size:]

        x1_sample = X1[sample_idx]
        x2_sample = X2[sample_idx]
        p = X1[test_idx].T
        q = X2[test_idx]

        # Estimate Essential matrix from the sample
        E = least_squares_estimation(x1_sample, x2_sample)

        e3_hat = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        residuals = []

        for i in range(len(q)):
            d1 = ((q[i] @ E @ p[:, i]) ** 2) / (np.linalg.norm(e3_hat @ E @ p[:, i]) ** 2)
            d2 = ((p.T[i] @ E.T @ q[i]) ** 2) / (np.linalg.norm(e3_hat @ E.T @ q[i]) ** 2)
            residuals.append(d1 + d2)

        inliers = [test_idx[i] for i, r in enumerate(residuals) if r < eps]
        if len(inliers) > best_num_inliers:
            best_num_inliers = len(inliers)
            best_E = E
            best_inliers = np.append(sample_idx, inliers)

    return best_E, best_inliers
