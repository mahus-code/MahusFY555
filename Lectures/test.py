import numpy as np
import matplotlib.pyplot as plt

# Define parameters
E_g = 1.17  # Bandgap in eV
t1 = -3.3  # Hopping parameter 1 in eV
t2 = t1 - E_g  # Hopping parameter 2 derived from bandgap

# Lattice and Hamiltonian setup
N = 100  # Number of sites (chain length)
a = 1.0  # Lattice constant

# Create the SSH Hamiltonian
def create_ssh_hamiltonian(N, t1, t2):
    H = np.zeros((2 * N, 2 * N))
    for i in range(N):
        # Couplings within unit cells
        H[2 * i, 2 * i + 1] = t1
        H[2 * i + 1, 2 * i] = t1

        # Couplings between unit cells
        if i < N - 1:
            H[2 * i + 1, 2 * (i + 1)] = t2
            H[2 * (i + 1), 2 * i + 1] = t2
    return H

# Compute energy spectrum and eigenstates
H = create_ssh_hamiltonian(N, t1, t2)
energies, eigenstates = np.linalg.eigh(H)

# Plot the energy spectrum
plt.figure(figsize=(8, 6))
plt.plot(range(2 * N), energies, 'o', label='Energy levels')
plt.axhline(0, color='black', linestyle='--', linewidth=0.7, label='Fermi level')
plt.xlabel("State Index")
plt.ylabel("Energy (eV)")
plt.title("Energy Spectrum of SSH Model")
plt.legend()
plt.grid()
plt.show()

# Visualize edge states if present
def plot_edge_states(eigenstates, N):
    edge_states = np.abs(eigenstates[:, np.argsort(np.abs(energies))[:2]]) ** 2
    x = np.arange(2 * N)

    plt.figure(figsize=(10, 6))
    for i in range(2):
        plt.plot(x, edge_states[:, i], label=f'Edge State {i+1}')

    plt.xlabel("Site Index")
    plt.ylabel("Probability Density")
    plt.title("Edge State Probability Distribution")
    plt.legend()
    plt.grid()
    plt.show()

plot_edge_states(eigenstates, N)
