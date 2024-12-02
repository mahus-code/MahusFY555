import numpy as np

from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target


from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, stratify = y, random_state = 1)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

pipe = Pipeline( steps = [("pca", PCA(n_components = 2)), ("scaler", StandardScaler()), ("classifier", Perceptron())] )

param_grid = [
    { 'classifier__eta0': np.linspace(0.00001, 1, 10),
     'classifier__max_iter': list(range(1,100,10))   
    } 
]

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(pipe, param_grid = param_grid, verbose = True, cv = 10, n_jobs = -1)

gs.fit(x_train, y_train)
print("best score:", gs.best_score_)
print("Best hyper parameter values:", gs.best_estimator_.get_params())


