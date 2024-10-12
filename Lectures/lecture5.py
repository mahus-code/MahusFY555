import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.integrate import odeint


# Differential eq:

def exponential_decay(t, y):
    return -0.5*y

# Solve:

sol = solve_ivp(exponential_decay, [0, 10], [1])
plt.plot(sol.t, sol.y.T, label = 'exponential decay')
#plt.legend(loc = 'best')
#plt.tight_layout()
#plt.show()

# New differential equation:
def exp_decay_odeint(y,t):
    return -0.5*y

# solve ODE:
t_interval = np.linspace(0, 10, 100)
sol2 = odeint(exp_decay_odeint, [1], t_interval)
plt.plot(t_interval, sol2, label= 'odeint')
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()

