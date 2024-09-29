import numpy as np
import pandas as pd

# pos = np.random.rand(3,3)
pos = np.array(((1,2,3), (4,5,6), (7,8,9)))
x = pos[:,0:1]
print(x)
print('----------------')
print(x.T)
print('----------------')
print(x.T-x)
print('----------------')
dx = np.zeros((3,3))
for i in range(3):
    # starts at i = 0 (first row)
    for j in range(3):
        # starts at j = 0 (first column)
        dx[i, j] = x[j, 0] - x[i, 0]

print(dx)
