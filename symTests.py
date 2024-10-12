import numpy as np
from sympy import symbols, solve
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random as rand

# Constants
t0 = 8.9  # Gyr
cMpcGyr = 306.3912716  # c in Mpc/Gyr

# Scale factor and Hubble parameter
def a(t):
    return (t/t0)**(2/3)

def H(t):
    return 2 / (3 * t)

# Geodesic equation system
def diffEquations(stateVarVector, lam_vals):
    # Unpack state variables: t, x, y, z, kt, kx, ky, kz
    t, x, y, z, kt, kx, ky, kz = stateVarVector
    
    # Hubble parameter
    H_t = H(t)
    
    # Geodesic equations
    dkt_dlambda = -H_t * kt**2
    dkx_dlambda = -2 * H_t * kx * kt
    dky_dlambda = -2 * H_t * ky * kt
    dkz_dlambda = -2 * H_t * kz * kt
    
    # Positions evolve based on velocities
    dt_dlambda = kt
    dx_dlambda = kx
    dy_dlambda = ky
    dz_dlambda = kz
    
    # Return the derivative vector
    return [dt_dlambda, dx_dlambda, dy_dlambda, dz_dlambda, dkt_dlambda, dkx_dlambda, dky_dlambda, dkz_dlambda]

# Initial conditions setup
def initParamInitialConditions():
    np.random.seed(17)
    # Initial positions set to 0 (origin)
    x = y = z = 0  
    
    # Initial time component for light-like geodesic
    ktic = -1 / cMpcGyr  # Choose a negative kt to go backward in time
    
    # Randomly choose kx and ky, and solve for kz
    kx = rand.uniform(0, 0.5)
    ky = rand.uniform(0, 0.5)
    kz_sym = symbols('kz')

    # Light-like geodesic constraint
    eq = (-cMpcGyr**2) * (ktic**2) + (a(t0)**2) * (kx**2 + ky**2 + kz_sym**2)
    kz_sol = solve(eq)
    kz = float(kz_sol[1])  # Choose the positive solution for kz
    
    # Return the initial state vector
    stateVar = [t0, x, y, z, ktic, kx, ky, kz]
    return stateVar, ktic

# Main solver and plotting
def main():
    # Initialize parameters and initial conditions
    stateVar, ktic = initParamInitialConditions()
    
    # Define the range of lambda values (affine parameter)
    lambdaValues = np.linspace(0, 1000, 1000)  # Larger range to capture more evolution
    
    # Solve the system of ODEs
    sol = odeint(diffEquations, stateVar, lambdaValues)

    # Extract solutions for plotting
    t_sol = sol[:, 0]  # Time
    x_sol = sol[:, 1]  # X coordinate
    y_sol = sol[:, 2]  # Y coordinate
    z_sol = sol[:, 3]  # Z coordinate
    kt_sol = sol[:, 4]  # Time component of 4-velocity

    # Redshift calculations
    z_direct = (1 / a(t_sol)) - 1
    z_ODE = (kt_sol / ktic) - 1

    # Plot the redshifts
    plt.plot(t_sol, z_direct, label=r'$z_{\text{direct}} = \frac{1}{a} - 1$', color='blue')
    #plt.plot(t_sol, z_ODE, label=r'$z_{\text{ODE}} = \frac{k^t}{k^t_{\text{ic}}} - 1$', color='green')
    plt.xlabel('Time (Gyr)')
    plt.ylabel('Redshift')
    plt.title('Redshift Evolution over Time')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
