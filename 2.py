import numpy as np
import matplotlib.pyplot as plt


# Define the MUSIC function
def MUSIC(M, N, sigma_s):
    sigma_w = 1
    theta_k = np.radians(np.array([0, 30, -60]))  # Convert degrees to radians
    K = len(theta_k)
    A = np.zeros((M, K), dtype=complex)
    for m in range(M):
        for k in range(K):
            A[m, k] = np.exp(1j * np.pi * np.sin(theta_k[k]) * m)

    X = np.sqrt(sigma_s) * np.random.randn(K, N)  # Source signal sequence
    W = np.sqrt(sigma_w) * np.random.randn(M, N)  # Noise
    Y = A @ X + W  # Signal + noise
    R = Y @ Y.conj().T / N  # Correlation matrix
    _, V = np.linalg.eigh(R)  # Eigen-decomposition of R
    V_2 = V[:, :M - K]  # Noise subspace

    theta_x = np.linspace(-100, 100, 200001)  # Fine angle sweep
    pseudo_spectrum = np.zeros(theta_x.shape)
    for i, theta in enumerate(theta_x):
        alpha = np.exp(1j * np.pi * np.sin(np.radians(theta)) * np.arange(M))
        pseudo_spectrum[i] = 1 / np.linalg.norm(V_2.conj().T @ alpha) ** 2

    pseudo_spectrum = 10 * np.log10(pseudo_spectrum / np.max(pseudo_spectrum))  # Convert to dB
    return theta_x, pseudo_spectrum


# Plot setup
plt.figure(figsize=(10, 6))
for i, params in enumerate([(10, 500, 10), (10, 50000, 10), (30, 500, 10), (10, 500, 100)], start=1):
    theta_x, pseudo_spectrum = MUSIC(*params)
    plt.plot(theta_x, pseudo_spectrum, linewidth=1.5,
             label=f'M={params[0]}, N={params[1]}, sigma_s^2={params[2]}')

plt.legend()
plt.grid(True)
plt.xlabel('Theta')
plt.ylabel('Pseudo Spectrum (dB)')
plt.axis([-100, 100, -70, 20])
plt.show()
