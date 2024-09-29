import numpy as np
import lec2_myfunctions

x = np.array([1, 2, 3])
M = np.array([ [4,5,6], [7,8,9] ])
ThreeD = np.array([ [[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]] ])

print("Matrix multiplication with arrays: ", M@x)

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.empty(3)

# i'th element of  should be ith element of a times i'th element of b
'''this is also a comment'''
'''
for i in range(len(a)):
    c[i] = a[i]*b[i]

d = np.empty(3)
d = a*b
print("c = ", c)
print("d = ", d)
'''
'''
print("Broadcasting multiplication: ", M*x)
'''

x = np.array([1, 2, 3])
xx = np.array([[1,2,3]])
print("Dimension of x: ", np.ndim(x))
print("Dimension of xx: ", np.ndim(xx))

print("x-x.T=", x-x.T)
print("xx-xx.T", xx-xx.T)

print(np.exp(x))


print(lec2_myfunctions.my_func1(2))