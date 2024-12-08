import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from sklearn.preprocessing import OneHotEncoder # not necessary; can just use pd but included here to show sklearn has it as well
# The following is largely based on https://jaketae.github.io/study/neural-net/

np.random.seed(0)



def sigmoid(x): # popular activation function
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(y_true, y_predict): # popular loss/cost function
    '''L(y_true, y_predict) = −∑y_true*log(y_predict).'''
    l_sum = np.sum(np.multiply(y_true, np.log(y_predict)))
    m = y_true.shape[0]
    l = -(1./m) * l_sum
    return l

class ThreeLayerNN:
    def __init__(self, N_features, N_data, N_hidden, N_classes):
        # Number of neurons for each layer:
        self.N_features = N_features
        self.N_data = N_data
        self.N_hidden = N_hidden
        self.N_classes = N_classes
        # Initialisation of parameters W1, W2, B1, B2:
        self.W1 = np.random.rand(N_features, N_hidden)
        self.B1 = np.random.rand(N_hidden)
        self.W2 = np.random.rand(N_hidden, N_classes)
        self.B2 = np.random.rand(N_classes)
    
    # Method to train our model
    def fit(self, X_train, Y_train, learning_rate=0.005, epochs=10000):
    # Epoch loop
        for epoch in range(epochs):
            error = 0
            # Training set Loop     
            """ Forward Propagation"""
            A1 = X_train @ self.W1 + self.B1
            z1 = sigmoid(A1)
            A2 = z1 @ self.W2 + self.B2
            z2 = sc.special.softmax(A2, axis=1)      
            """The output of softmax is a vector containing the propabilities of belonging to each category."""
                    
            # Loss:
            error = cross_entropy_loss(Y_train, z2)# Using cross entropy as loss function is a typical choice
            # "error" is the error function we use, but note that it is not used explicitly
            # - we only need its derivatives in the gradient descent.
            # Often it is computed anyway and printed out but I have ommitted the printput here.
            
            """ Remember here that both Z2 and y_train are matrices we can think of as representing
            the probability of being each of the 3 iris types."""
            
            """ Backpropagation"""
            delta2 = z2 - Y_train # dL/dz2
            delta1 = (delta2).dot(self.W2.T) * z1 * (1 - z1) # dL/dz1
 
            # Gradient descent:
            self.W2 -= learning_rate * z1.T.dot(delta2)
            self.B2 -= learning_rate * (delta2).sum(axis=0)

            self.W1 -= learning_rate * X_train.T.dot(delta1)
            self.B1 -= learning_rate * (delta1).sum(axis=0)

    def predict(self, X_test): # for testing data once the weights have been computed using the fit method above
        # Forward Propagation
        A1 = X_test @ self.W1 + self.B1
        z1 = sigmoid(A1)
        A2 = z1 @ self.W2 + self.B2
        z2 = sc.special.softmax(A2, axis =1)   
        return z2
    
# Dataset
N_features = int(2)
N_data = int(154) # number of data points in training set 
N_classes = int(3) # number of nodes in output layer = number of categories
N_hidden = int(np.floor((N_features+N_data)/2)) # number of nodes in hidden layer
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
one_hot = pd.get_dummies(df) # Get one-hot encoding of variable
df2 = df.merge(one_hot) # Join the one hot encoding to the data frame
df3 = df2.drop(4,axis = 1) # Drop the 4th column which contians the category labels

Y_train = df3.iloc[0:N_data,4:7].values # we extract the first N_data class labels in one-hot encoding, store them using numpy representation
X_train = df3.iloc[0:N_data, [0,2]].values # want N_data first rows of column 0 and 2

Y_test = df3.iloc[N_data:, 4:7].values
X_test = df3.iloc[N_data:, [0,2]].values
print('Test data:',X_test, Y_test)

# Network object creation and training

fnn = ThreeLayerNN(N_features, N_data, N_hidden, N_classes)
fnn.fit(X_train, Y_train)

