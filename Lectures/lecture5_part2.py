from scipy.integrate import quad
import numpy as np
from scipy import linalg # np.linalg
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-x*x)

i = quad(f, 0, 1)
print(i)

A = np.array([[1,2], [1,1]])
b = np.array([0,1])

x = linalg.solve(A, b)
print(x)

# mock data:
x =np.array([0, 1.2, 3, 4, 5, 6, 7])
y = np.array([-1, 0.2, 0.9, 2.1, 3, 2.8, 4.2])

# Set up matrix stuff:
ones = np.ones(len(x))
A = np.vstack([x, ones]).T

#print(A)

# Find best fit:
k, rs, rank, s = linalg.lstsq(A, y)

print(k)
plt.plot(x,y,'.')
plt.plot(x, k[0]*x + k[1], '-')
plt.tight_layout()
plt.show()