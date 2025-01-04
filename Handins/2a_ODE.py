import numpy as np
from sympy import diff, symbols, solve, lambdify
import random as rand
from scipy.integrate import odeint
import matplotlib.pyplot as plt



t0 = 8.9 # [Gyr]

# Define our a(t) symbolically
t = symbols('t')
a_sym = (t/t0)**(2/3)

# find da/dt
a_diff = diff(a_sym, t)

# convert back into a lambda function
aDiff = lambdify(t, a_diff, "numpy")

def a(t): # function for our a(t)
    return (t/t0)**(2/3)

def H(t): # function for our H(t), using a(t) and previous lambda function
    return aDiff(t)/a(t)

def zDirect(t): # zDirect function to check redshift
    return (1/a(t)) - 1

def diffEquations(stateVarVector, lambda_val):
    '''
    Defines our coupled ODEs
    stateVarVector: vector of our state variables = t, x, y, z, kz, kx, ky, kt
    lambda_val: wavelength values
    '''
    
    # Unpack our state variable vector
    t, x, y, z, kt, kx, ky, kz = stateVarVector

    # Calculate our derivatives
    dkt_dlambda = -H(t)*(kt**2) # kt'
    dkx_dlambda = -2*H(t)*kx*kt # kx'
    dky_dlambda = -2*H(t)*ky*kt # ky'
    dkz_dlambda = -2*H(t)*kz*kt # kz'

    # Vector of derivatives = [t', x', y', z', kt', kx', ky', kz']
    derivVector = [kt, kx, ky, kz, dkt_dlambda, dkx_dlambda, dky_dlambda, dkz_dlambda]

    return derivVector

def initParamInitialConditions(): # function to initialize our state vector and initial conditions
    # Parameters
    cMpcGyr = 306.3912716
    
    # Initial conditions
    a = 1
    ktic = -1/cMpcGyr # kt initial value
    
    # Choose random kx, and ky
    np.random.seed(17)
    kx = rand.uniform(0, 0.5)
    ky = rand.uniform(0, 0.5)

    # Solve for kz symbolically using defined relationship
    kz = symbols('kz')
    eq = (-cMpcGyr**2) * (ktic**2) + (a**2) * ( (kx**2) + (ky**2) + (kz**2) )
    kz = solve(eq)

    # Ensure that we choose a positive kz (although it might work with < 0)
    for value in kz:
        if value > 0:
            kz = value
            break
    x=0
    y=0
    z=0
    
    stateVar = [t0, x, y, z, ktic, kx, ky, kz]
    return stateVar, ktic

def main() -> None:
    stateVar, ktic = initParamInitialConditions() # call the initial conditions and assign to stateVar and ktic

    lambdaValues = np.linspace(0, 1500, 1000) # generate lambda values
    sol = odeint(diffEquations, stateVar, lambdaValues) # pass stateVar and wavelength array
    
    kt_sol = sol[:, 4] # 5th column has the kt solutions
    t_sol = sol[:, 0] # 1st colum contains the time

    # Initialize the figure for the plot
    figure, (ax1, ax2) = plt.subplots(2, 1)

    # Plot kt/kt_initial - 1 (ODE) versus time 
    ax1.plot(t_sol, kt_sol/ktic - 1) 
    ax1.set_title(r'Redshift calculated from ODE solution $\left(\frac{k^t}{k^t_{ic}}-1\right)$')
    ax1.set_ylabel(r'Redshift $(z_{ODE})$')
    ax1.set_xlabel('Time [Gyr]')
    ax1.grid(True)
    
    # Plot 1/a(t)-1 versus time (z_direct)
    ax2.plot(t_sol, zDirect(t_sol), color='red') # plot (z-direct)
    ax2.set_title(r'Redshift calculated directly from a(t) $\left(\frac{1}{a(t)}-1\right)$')
    ax2.set_ylabel(r'Redshift $(z_{direct})$')
    ax2.set_xlabel('Time [Gyr]')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
