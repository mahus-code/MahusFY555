from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target

from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transform(x)

# Make covariance matrix:

cov_mat = np.cov(x_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('Eigen vectors:', eigen_vecs)
print('Eigen values:', eigen_vals)

# Variance explained ratio
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse = True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(np.arange(len(var_exp)), var_exp, alpha = 0.5, align='center', label='Individual explained variance')

plt.step(np.arange(len(cum_var_exp)), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

# Make projection matrix:
eigen_pairs = [ (eigen_vals[i], eigen_vecs[:,i]) for i in range(len(eigen_vals)) ]
eigen_pairs.sort(key= lambda k: k[0], reverse=True)

W = np.hstack(( eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:,np.newaxis] ))

x_pca = x_std.dot(W)

colors = ['r', 'g', 'b']
markers = ['o', 'x', 's']

for l,c, m in zip(np.unique(y), colors, markers):
    plt.scatter(x_pca[y==l, 0], x_pca[y==l, 1], color=c, label=l, marker=m)

plt.xlabel('pc 1')
plt.ylabel('pc 2')
plt.legend(loc='best')
plt.show()