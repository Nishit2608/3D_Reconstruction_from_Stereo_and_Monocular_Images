import matplotlib.pyplot as plt
import numpy as np

def plot_lines(lines, h, w):
    for i in range(lines.shape[1]):
        if abs(lines[0, i] / lines[1, i]) < 1:
            y0 = -lines[2, i] / lines[1, i]
            yw = y0 - w * lines[0, i] / lines[1, i]
            plt.plot([0, w], [y0, yw])
        else:
            x0 = -lines[2, i] / lines[0, i]
            xh = x0 - h * lines[1, i] / lines[0, i]
            plt.plot([x0, xh], [0, h])

def plot_epipolar_lines(image1, image2, uncalibrated_1, uncalibrated_2, E, K, plot=True):
    epipolar_lines_in_1 = []
    epipolar_lines_in_2 = []

    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv  # Fundamental matrix

    for i in range(uncalibrated_1.shape[1]):
        epipolar_lines_in_2.append(F @ uncalibrated_1[:, i])
        epipolar_lines_in_1.append(F.T @ uncalibrated_2[:, i])

    epipolar_lines_in_1 = np.array(epipolar_lines_in_1).T
    epipolar_lines_in_2 = np.array(epipolar_lines_in_2).T

    if plot:
        plt.figure(figsize=(16, 6))
        h, w = image1.shape[:2]

        plt.subplot(1, 2, 1)
        plt.imshow(image1[..., ::-1])
        plt.title("Epipolar Lines in Image 1")
        plot_lines(epipolar_lines_in_1, h, w)

        plt.subplot(1, 2, 2)
        plt.imshow(image2[..., ::-1])
        plt.title("Epipolar Lines in Image 2")
        plot_lines(epipolar_lines_in_2, h, w)

        plt.tight_layout()
        plt.show()
