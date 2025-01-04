import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, LearningCurveDisplay, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# First we load the wine data set
wine = datasets.load_wine()
x = wine.data
y = wine.target

# Then we split the data and construct our pipeline
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, stratify=y, random_state=1)
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=4), LogisticRegression(max_iter=100, random_state=1))
pipe_lr.fit(xtrain, ytrain)

# We create a learning curve display using training sizes from 50 to 110 with 10 increments
# Cross validation fold is set to 5 - meaning the data set is split into 5 folds
LearningCurveDisplay.from_estimator(pipe_lr, x, y, train_sizes=[50, 60, 70, 80, 90, 100, 110], cv=5)
plt.ylabel('Accuracy score')
plt.title('Learning Curve Graph of Pipeline with StandardScaler, PCA and LogisticRegression Model \n for the wine dataset')
plt.grid(True)
plt.show()

# Prints the named steps and possible parameter keys
'''
print(pipe_lr.named_steps)
print(pipe_lr.get_params().keys())
'''

# Hyperparameters to tune:
param_grid = [
    { 'logisticregression__C': np.logspace(-4, 4, 10),
     'logisticregression__max_iter': list(range(100,1000,10)), 
     'logisticregression__solver': ['lbfgs','newton-cg', 'sag']   
    } 
]

# Create our randomized k-fold cross validation grid search
randGridSearch = RandomizedSearchCV(pipe_lr, param_distributions=param_grid, verbose=True, cv=10, n_jobs=-1)

randGridSearch.fit(xtrain, ytrain)

ypred_train = randGridSearch.predict(xtrain)
ypred_test = randGridSearch.predict(xtest)

print("Accuracy score of final model (result from k-fold cross validation grid search) on training data:", accuracy_score(ytrain, ypred_train))
print("Accuracy score of final model (result from k-fold cross validation grid search) on test data:", accuracy_score(ytest, ypred_test))
