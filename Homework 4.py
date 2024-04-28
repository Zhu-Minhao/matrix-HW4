import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# Define the MUSIC function
def MUSIC(M, N, sigma_w):
    sigma_s = 1
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
    spectrum = np.zeros(theta_x.shape)
    for i, theta in enumerate(theta_x):
        alpha = np.exp(1j * np.pi * np.sin(np.radians(theta)) * np.arange(M))
        spectrum[i] = 1 / np.linalg.norm(V_2.conj().T @ alpha) ** 2

    spectrum = 10 * np.log10(spectrum / np.max(spectrum))  # Convert to dB
    return theta_x, spectrum


# parameters = [(10, 500, 1), (10, 5000, 1), (30, 500, 1), (10, 500, 10)]
parameters = [(10, 500, 1)]
# Plot setup
plt.figure(dpi=100)
for i, params in enumerate(parameters, start=1):
    theta_x, pseudo_spectrum = MUSIC(*params)
    plt.plot(theta_x, pseudo_spectrum, linewidth=1.5, label=fr'$M={params[0]},\ N={params[1]},\ \sigma_w^2={params[2]}$')

plt.legend()
plt.grid(True)
plt.xlabel('Theta')
plt.ylabel('Spectrum (dB)')
plt.axis([-100, 100, -60, 10])
plt.show()
