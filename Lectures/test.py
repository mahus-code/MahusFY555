import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def ssh_chain(num_sites, t1, t2):
    """
    Generate an SSH Hamiltonian for a chain.

    Parameters:
        num_sites (int): Number of sites in the chain.
        t1 (float): Intracell hopping parameter.
        t2 (float): Intercell hopping parameter.

    Returns:
        np.ndarray: SSH Hamiltonian matrix.
    """
    H = np.zeros((num_sites, num_sites))
    for i in range(num_sites - 1):
        if i % 2 == 0:
            H[i, i + 1] = t1
        else:
            H[i, i + 1] = t2
        H[i + 1, i] = H[i, i + 1]
    return H

def planck_distribution(omega, T):
    """
    Compute the Planck distribution.

    Parameters:
        omega (float): Angular frequency.
        T (float): Temperature in Kelvin.

    Returns:
        float: Energy density at omega.
    """
    hbar = 1.0545718e-34  # Reduced Planck constant (J.s)
    kB = 1.380649e-23     # Boltzmann constant (J/K)
    if T == 0:
        return 0
    return hbar * omega / (np.exp(hbar * omega / (kB * T)) - 1)

def radiative_heat_transfer(chain1, chain2, T1, T2, freq_range):
    """
    Compute the radiative heat transfer between two chains.

    Parameters:
        chain1 (np.ndarray): Hamiltonian of chain 1.
        chain2 (np.ndarray): Hamiltonian of chain 2.
        T1 (float): Temperature of chain 1 (K).
        T2 (float): Temperature of chain 2 (K).
        freq_range (np.ndarray): Frequency range for integration.

    Returns:
        float: Total radiative heat transfer.
    """
    eigenvals1 = np.linalg.eigvalsh(chain1)
    eigenvals2 = np.linalg.eigvalsh(chain2)

    total_heat = 0
    for omega in freq_range:
        n1 = planck_distribution(omega, T1)
        n2 = planck_distribution(omega, T2)

        # Simplified radiative transfer calculation
        spectral_overlap = np.exp(-np.abs(omega - eigenvals1[:, None])) @ np.exp(-np.abs(omega - eigenvals2[None, :]))

        heat_flow = spectral_overlap.sum() * (n1 - n2)
        total_heat += heat_flow

    return total_heat

# Parameters
num_sites = 10
t1, t2 = 1.0, 0.5
T1, T2 = 300, 100  # Temperatures in Kelvin
freq_range = np.linspace(0.1, 10, 100)  # Frequency range in arbitrary units

# Construct SSH chains
chain1 = ssh_chain(num_sites, t1, t2)
chain2 = ssh_chain(num_sites, t2, t1)

# Compute radiative heat transfer
heat_transfer = radiative_heat_transfer(chain1, chain2, T1, T2, freq_range)
print(f"Radiative heat transfer: {heat_transfer:.2e} units")
