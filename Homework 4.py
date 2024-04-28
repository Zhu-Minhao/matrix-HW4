import numpy as np
import matplotlib.pyplot as plt


# MUSIC Algorithm Function
def MUSIC_gys(M, N, sigma_s):
    sigma_w = 1
    theta_k = np.array([0, 30, -60]) * np.pi / 180
    K = len(theta_k)
    A = np.zeros((M, K), dtype=complex)
    for m in range(M):
        for k in range(K):
            A[m, k] = np.exp(1j * np.pi * np.sin(theta_k[k]) * m)
    X = np.sqrt(sigma_s) * np.random.randn(K, N)
    W = np.sqrt(sigma_w) * np.random.randn(M, N)
    Y = A @ X + W
    R = Y @ Y.conj().T / N
    _, V = np.linalg.eig(R)
    V_2 = V[:, :M-K]
    theta_x = np.linspace(-100, 100, 200001)
    pseudo_spectrum = np.zeros(theta_x.shape)
    for i, theta in enumerate(theta_x):
        alpha = np.exp(1j * np.pi * np.sin(theta * np.pi / 180) * np.arange(M))
        pseudo_spectrum[i] = 1 / np.linalg.norm(V_2.conj().T @ alpha)**2
    return theta_x, 10 * np.log10(pseudo_spectrum / np.max(pseudo_spectrum))


# Parameters
N = 500
M = 10
sigma_s = 10
color_RGB = np.array([
    [0, 0, 0], [255, 153, 0], [255, 0, 102], [0, 102, 255], [204, 0, 255],
    [0, 102, 204], [255, 51, 0], [0, 153, 153], [204, 153, 0], [204, 204, 0], [109, 164, 58]
]) / 255


# Generate data using MUSIC algorithm
theta_x_1, pseudo_spectrum_1 = MUSIC_gys(10, 500, 10)
theta_x_2, pseudo_spectrum_2 = MUSIC_gys(10, 50000, 10)
theta_x_3, pseudo_spectrum_3 = MUSIC_gys(30, 500, 10)
theta_x_4, pseudo_spectrum_4 = MUSIC_gys(10, 500, 100)

# Plotting
plt.figure()
plt.plot(theta_x_1, pseudo_spectrum_1, label='M=10, N=500, sigma_s^2 = 1')
plt.plot(theta_x_2, pseudo_spectrum_2, label='M=10, N=50000, sigma_s^2 = 1')
plt.plot(theta_x_3, pseudo_spectrum_3, label='M=30, N=500, sigma_s^2 = 1')
plt.plot(theta_x_4, pseudo_spectrum_4, label='M=10, N=500, sigma_s^2 = 10')
plt.legend()
plt.grid(True)
plt.xlabel('Theta')
plt.ylabel('Pseudo Spectrum (dB)')
plt.axis([-100, 120, -70, 20])
plt.show()
