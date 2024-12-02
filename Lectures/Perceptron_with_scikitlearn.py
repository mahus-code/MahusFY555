import numpy as np
from sklearn import datasets

"""scikit-learn contains a number of data sets for machine learning, including the iris dataset"""
iris = datasets.load_iris()
x = iris.data[:, [1,3]] # use only 2 features (as we've done earlier)
y = iris.target
print('Class labels:', np.unique(y)) # print to see how iris types are stored

""" With scikit-learn it's easy to split data into a training and test set -- we didn't do this earlier, but let's do it now! """
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.3, random_state = 1, stratify=y)
""" Note that the data is shuffled randomly before splitting into 30 % test and 70 % train
    - with fixed random state we can reproduce our results. the stratification means that the 
    training and testing data sets have the same proportions of the different y-values (iris types)
"""

# We can also use scikit-learn to do feature scaling of our data:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(xtrain) # Estimates values of mean and standard deviation of data
xtrain_std = sc.transform(xtrain) # standardized training set
xtest_std = sc.transform(xtest) # standardizes test set


# Training a perceptron on the training data:
from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=100, alpha=0.05, random_state=1)
ppn.fit(xtrain_std,ytrain)

# Now we can make predictions of the test set:
ypred = ppn.predict(xtest_std)

from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(ytest,ypred)) # prints accuracy in percent


# This was realy easy, so let's make an extra effort and plot our results nicely!
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions( x, y, classifier, test_idx = None, resolution=0.02):
    markers = ('s', 'x', 'o')#, '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray') # 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:,0].min()-1, x[:,0].max()+1 # plus one and minus one gives a buffer region for plotting
    x2_min, x2_max = x[:,1].min()-1 , x[:,1].max()+1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)# predict classification corresponding to grid points
    z = z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl,0], y = x[y==cl,1], alpha = 0.8, marker = markers[idx], label=cl, edgecolor = 'black')

    if test_idx:
        xtest, ytest = x[test_idx,:], y[test_idx]
        plt.scatter(xtest[:,0], xtest[:,1], facecolor = 'None', edgecolor='black', marker='o', s = 100, label = 'test set')
        
xcombined_std = np.vstack((xtrain_std, xtest_std))
ycombined = np.hstack((ytrain,ytest))
plot_decision_regions(x=xcombined_std, y = ycombined, classifier=ppn, test_idx=range(140,150))
plt.xlabel('petal length (standardized)')
plt.ylabel('petal width (standardized)')
plt.legend(loc='upper left')
plt.show()