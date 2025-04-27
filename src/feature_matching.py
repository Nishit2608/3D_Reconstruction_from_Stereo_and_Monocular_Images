import cv2
import numpy as np

def detect_and_match_features(img1, img2):
    """
    Detect SIFT keypoints and match them using BFMatcher with Lowe's ratio test.
    Returns keypoints, descriptors, and good matched points (pts1, pts2).
    """
    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Brute Force matcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Get matching keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return kp1, kp2, good_matches, pts1, pts2
    