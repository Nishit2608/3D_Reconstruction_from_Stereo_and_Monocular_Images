import numpy as np

def reconstruct3D(transform_candidates, calibrated_1, calibrated_2):
    """
    Selects the best (R, T) candidate based on triangulated points being in front of both cameras.
    
    Args:
        transform_candidates: List of 4 possible (R, T) candidates
        calibrated_1: Nx3 array of normalized points from image 1
        calibrated_2: Nx3 array of normalized points from image 2
    
    Returns:
        P1: 3D points triangulated with camera 1
        P2: 3D points triangulated with camera 2
        Best T, Best R
    """
    best_num_front = -1
    best_candidate = None
    best_lambdas = None

    for candidate in transform_candidates:
        R = candidate['R']
        T = candidate['T']

        lambdas = np.zeros((2, calibrated_1.shape[0]))

        for j, (point1, point2) in enumerate(zip(calibrated_1, calibrated_2)):
            point1 = point1.reshape(3, 1)
            point2 = point2.reshape(3, 1)

            A = np.column_stack((point2, -(R @ point1)))
            B = T.reshape(3, 1)

            # Solve for scale factors
            lambda_values = np.linalg.inv(A.T @ A) @ A.T @ B
            lambdas[0, j] = lambda_values[0]
            lambdas[1, j] = lambda_values[1]

        # Count how many points are in front of both cameras
        num_front = np.sum(np.logical_and(lambdas[0] > 0, lambdas[1] > 0))

        if num_front > best_num_front:
            best_num_front = num_front
            best_candidate = candidate
            best_lambdas = lambdas

    # Triangulate 3D points
    P1 = best_lambdas[1].reshape(-1, 1) * calibrated_1
    P2 = best_lambdas[0].reshape(-1, 1) * calibrated_2

    T = best_candidate['T']
    R = best_candidate['R']
    
    return P1, P2, T, R
