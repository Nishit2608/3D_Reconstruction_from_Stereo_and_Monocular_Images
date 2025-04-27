import numpy as np

def least_squares_estimation(X1, X2):
    """
    Estimates the Essential Matrix using the normalized 8-point algorithm (least squares).
    
    Args:
        X1: N x 3 array of points from image 1 (homogeneous coordinates)
        X2: N x 3 array of points from image 2 (homogeneous coordinates)
    
    Returns:
        E: Essential Matrix (3x3)
    """
    n = len(X1)
    A = np.zeros((n, 9))

    for i in range(n):
        A[i, 0:3] = X1[i, 0] * X2[i, :]
        A[i, 3:6] = X1[i, 1] * X2[i, :]
        A[i, 6:9] = X1[i, 2] * X2[i, :]

    # Solve for E using SVD
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)

    # Enforce the internal constraint that two singular values are equal and one is zero
    U, S, Vt = np.linalg.svd(E)
    S = np.diag([1, 1, 0])
    E = U @ S @ Vt

    return E
