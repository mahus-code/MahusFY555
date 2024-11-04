import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron():
    def __init__(self, eta = 0.01, n_iter = 10, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self,x): # compute z. For adaline, this is the activation function
        return np.dot(x, self.w[1:])+self.w[0]
    
    def predict(self,x): # give class label. This is the decision function
        return np.where(self.net_input(x)>=0.0,1,-1)
    
    def fit(self, x,y): # This is the learning rule, also called "fitting", where we update the values of w
        # (x has dimensions [n_samples, n_features], y has dimension [n_sample])
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc = 0.0, scale = 0.1, size = 1+x.shape[1])
        
        self.errors = []
        for i in range(self.n_iter):
            errors_ = 0
            for xi, target in zip(x,y):
                update = self.eta*(target-self.predict(xi))
                self.w[1:]+=update*xi
                self.w[0]+=update
                
                errors_+=int(update!=0.0)
            self.errors.append(errors_)
        return self
    
# End og Perceptron class!

''' Now we can import a dataset and fit weights. We will use the iris dataset which has 150 data points 
    representing three types of iris flowers. The first 100 data points correspond to only two types of flowers
    and we will only be using the first 100 datapoints for now. '''
'''
#Import data from webpage:
#(Not working at right this moment, presumably the webpage is down but will be up again soon -- go to alternative below)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.head()

y=df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa', 1,-1)
x = df.iloc[0:100,[0,2]].values
print(x.shape)
'''
# Alternative for importing data: Luckily we are using data that is available through sklearn which we'll talk about next week.
# For now, we can import data using sklearn since the webpage above is sometimes down...
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data[0:100, [0,2]] # Pulls out the first 100 data points, only taking features 0 and 2 (out of 4)
print("x", x)


y = iris.target[0:100] # Puls out the first 100 data points, taking the target.
print(y)
y = np.where(y==0, 1,-1) # Renames from string to 1 or -1.
print(y)
#y = np.where(y==0, 1, -1)
print(y)
print(y.shape)

# Plot of data 
plt.scatter(x[:50,0], x[:50,1], marker='o', color='red', label='Setosa')
plt.scatter(x[50:100,0], x[50:100,1], marker='x', color='blue', label='Versicolor')

plt.title('Scatterplot of data')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.legend(loc='upper left')
plt.show()


ppn = Perceptron(eta = 0.01, n_iter = 10, random_state = 10)
ppn.fit(x,y)
plt.plot(range(1, len(ppn.errors)+1), ppn.errors, marker='o')
plt.title('Error')
plt.xlabel('epochs')
plt.ylabel('number of wrong categorizations')
plt.show()


