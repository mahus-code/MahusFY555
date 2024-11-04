import numpy as np

from sklearn import datasets

iris = datasets.load_iris()

x = iris.data[:, [1,3]]
y = iris.target

print('Class labels:', np.unique(y)) # We don't want 0, 1 and 2 --> "one hard" coding

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.3, random_state = 1, stratify = y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(xtrain) # find mu and sigma
xtrain_std = sc.transform(xtrain)
xtest_std = sc.transform(xtest)

# Training the perceptron:
from sklearn.linear_model import Perceptron
percy =  Perceptron(max_iter = 10, eta0 = 0.05, random_state = 1)
percy.fit(xtrain_std, ytrain)
ypred = percy.predict(xtest_std)

from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(ytest, ypred))
# We now plot it to get further understanding

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(x,y, classifier, test_idx = None, resolution = 0.02):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'green', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Make boundaries for regions
    x1_min, x1_max = x[:,0].min() - 1, x[:,0].max()+1
    x2_min, x2_max = x[:,1].min() - 1, x[:,1].max()+1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2,z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x[y==cl, 0], x[y==cl, 1], alpha = 0.8, marker = markers[idx], label = cl, edgecolors= 'black')

    if test_idx:
        xtest, ytest = x[test_idx, :], y[test_idx]
        plt.scatter(xtest[:,0], xtest[:,1], facecolor='None', edgecolors='black', marker='o', s=100, label='test set')


xcombined_std = np.vstack((xtrain_std, xtest_std))
ycombined_std = np.hstack((ytrain, ytest))

plot_decision_regions(xcombined_std, ycombined_std, percy, test_idx=range(140, 150))

plt.xlabel('standardized feature 1')
plt.ylabel('standardized feature 2')
plt.legend(loc = 'upper left')
plt.show()