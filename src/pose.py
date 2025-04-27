import numpy as np

def pose_candidates_from_E(E):
    """
    Decompose the Essential Matrix E into four possible (R, T) transformation pairs.
    Returns a list of 4 candidate dictionaries with keys 'R' and 'T'.
    """
    R_90_pos = np.array([[0, -1, 0],
                         [1,  0, 0],
                         [0,  0, 1]])
    
    R_90_neg = np.array([[0,  1, 0],
                         [-1, 0, 0],
                         [0,  0, 1]])

    # SVD decomposition of E
    U, S, Vt = np.linalg.svd(E)

    # Two possible translations
    T1 = U[:, 2]
    T2 = -T1

    # Two possible rotations
    R1 = U @ R_90_pos.T @ Vt
    R2 = U @ R_90_neg.T @ Vt

    # Four possible (R, T) combinations
    return [
        {"R": R1, "T": T1},
        {"R": R2, "T": T1},
        {"R": R1, "T": T2},
        {"R": R2, "T": T2}
    ]
