import numpy as np


test_1 = np.arange(5)
print(test_1)

x = np.array([[1, 0], [0, 0], [1, 1], [0, 1], [0,1]])
print(x.shape)

z = tuple(zip(x, test_1))


myList = []
for xi, target in z:
    myList.append(xi)

print(myList)

rgen = np.random.RandomState(1)
w = rgen.normal(loc=0.0, scale=0.1, size=3)
print(w)
print(w[1:].shape)
xi = np.array([2, 1])
print("dot", np.dot(xi, w[1:]))
