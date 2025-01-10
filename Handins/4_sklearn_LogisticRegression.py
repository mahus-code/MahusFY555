import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

x = iris.data[50:, [1,2]] # Exclude the first 50 data points, and use feature 2 and 3
y = iris.target[50:] # Exclude the first 50 target data points

print('Class labels:', np.unique(y)) # We don't want 0, 1 and 2 --> "one hard" coding

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.3, random_state = 1, stratify = y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(xtrain) # find mu and sigma
xtrain_std = sc.transform(xtrain)
xtest_std = sc.transform(xtest)

from sklearn.linear_model import LogisticRegression
Lr =  LogisticRegression(C = 1, max_iter = 7, random_state = 1)
Lr.fit(xtrain_std, ytrain)
ypred = Lr.predict(xtest_std)

Lr_nonStandard = LogisticRegression(C=1, max_iter=7, random_state=1)
Lr_nonStandard.fit(xtrain, ytrain)
nonStd_pred = Lr_nonStandard.predict(xtest)
print("------------------------------------------------------------------------------------")
print("Hyper parameters used for Logistic Regression model are C, max_iter, and random_state")
print("Where C is the inverse regularization strength and we use a default value.", 
      "\nWe use max_iter and set it to 100 to ensure the model is sufficiently trained",
       "and random_state to be able to replicate our results.")
print("------------------------------------------------------------------------------------")

from sklearn.metrics import accuracy_score
print('Accuracy for standardized training data (7 iterations):', accuracy_score(ytest, ypred))
print('Accuracy of non-standardized training data (7 iterations):', accuracy_score(ytest, nonStd_pred))
print("------------------------------------------------------------------------------------")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

''' 
Calculate the confusion matrices of both the regular logistic regression and the non-standardized data
and display the plots.
'''
cm_std = confusion_matrix(ytest, ypred, labels=Lr.classes_)
disp_cm_std = ConfusionMatrixDisplay(confusion_matrix=cm_std, display_labels=Lr.classes_)
disp_cm_std.plot()
plt.title("Confusion Matrix for Logistic Regression model trained with standardized data")

cm_non_std = confusion_matrix(ytest, nonStd_pred, labels=Lr_nonStandard.classes_)
disp_cm_nonStd = ConfusionMatrixDisplay(confusion_matrix=cm_non_std, display_labels=Lr_nonStandard.classes_)
disp_cm_nonStd.plot()
plt.title("Confusion Matrix for Logistic Regression model trained with non-standardized data")

plt.show()

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

    if test_idx: # if test_idx is set
        x_train, y_train = np.delete(x, test_idx, axis=0), np.delete(y, test_idx, axis=0) # create xtrain and ytrain where
        # we delete the data indexed by 'test_idx'
        print("x",x_train)
        
        for idx, cl in enumerate(np.unique(y)): # Plot the remaining data
            print("y",y_train==cl)
            plt.scatter(x_train[y_train==cl, 0], x_train[y_train==cl, 1], alpha = 0.8, marker = markers[idx], label = cl, edgecolors= 'black')

    if test_idx: # Then we plot the test portion
        xtest, ytest = x[test_idx, :], y[test_idx]
        plt.scatter(xtest[:,0], xtest[:,1], facecolor='None', edgecolors='black', marker='o', s=100, label='test set')


xcombined_std = np.vstack((xtrain_std, xtest_std))
ycombined_std = np.hstack((ytrain, ytest))

plot_decision_regions(xcombined_std, ycombined_std, Lr, test_idx=range(70, 100))

plt.xlabel('standardized feature 1')
plt.ylabel('standardized feature 2')
plt.legend(loc = 'upper left')
plt.show()

