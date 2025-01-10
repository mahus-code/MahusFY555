import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import scipy as sc
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
 

np.random.seed(17)
# Function to create the coordinates of the nodes (circles) 
def circle_maker(circles, N_features, N_hidden, N_classes):
    for i in range(N_features):
        if i == 0:
            circles.append((i,-6))
        elif (i > 0 and i < 7):
            circles.append((-i*10, -6))
        else:
            circles.append((i*10-6*10, -6))

    for j in range(N_hidden):
        if j == 0:
            circles.append((0.5*2, 45))
        elif (j > 0 and j < 34):
            circles.append((j*2+0.5*2, 45))
        elif (j==34):
            circles.append((-0.5*2, 45))
        else:
            circles.append((34.5*2-j*2, 45))

    for k in range(N_classes):
        if k == 0:
            circles.append((0.0, 75))
        if k == 1:
            circles.append((-45.0, 75))
        if k==2:
            circles.append((45.0, 75))
    return None

def sigmoid(x): # popular activation function
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(y_true, y_predict): # popular loss/cost function
    '''L(y_true, y_predict) = −1/m ∑y_true*log(y_predict).'''
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
# -------------------------------------------------------------------------------------------------------------
# Load our dataset
data = datasets.load_wine()
X = data['data']
y = data['target']
y = y.reshape(-1,1) # Reshape into 2D array for fit_transform   
encoder = OneHotEncoder(sparse_output=False)
one_hot_y = encoder.fit_transform(y)

# Split the data
xtrain, xtest, ytrain, ytest = train_test_split(X, one_hot_y, test_size=0.3, random_state=1, stratify=one_hot_y)

N_features = int(xtrain.shape[1]) # gives us 13 features
N_data = int(xtrain.shape[0]) # number of data points in training set, 124 (Total for wine is 178)
N_classes = int(3) # number of nodes in output layer = number of categories
N_hidden = int(np.floor((N_features+N_data)/2)) # number of nodes in hidden layer (68)

# -------------------------------------------------------------------------------------------------------------

# Network object creation and training
fnn = ThreeLayerNN(N_features, N_data, N_hidden, N_classes)
fnn.fit(xtrain, ytrain)

# -------------------------------------------------------------------------------------------------------------

# Normalize the weights
sc = StandardScaler()
weight1_sc = sc.fit_transform(fnn.W1)
weight2_sc = sc.fit_transform(fnn.W2)

# Create our tuple array with x,y coordinates for the nodes (circles)
circles = []
circle_maker(circles, N_features=N_features, N_hidden=N_hidden, N_classes=N_classes)

# Extract the three layers of nodes - sorted in ascending order (of x-coordinate)
nodes_input = circles[0:N_features]
nodes_input_ = sorted(nodes_input, key=lambda x: x[0]) # lambda key specifies first element of tuple to sort after

nodes_hidden = circles[13:81]
nodes_hidden_ = sorted(nodes_hidden, key=lambda x: x[0]) 

nodes_output = circles[81::]
nodes_output_ = sorted(nodes_output, key=lambda x: x[0])

# Generate plot
fig, ax = plt.subplots(figsize=(15, 8))

'''Draw the circles, based on coordinates found earlier'''
for circle in circles:
    circ = patches.Circle(circle, radius=0.55, color='black')
    ax.add_patch(circ)

# -------------------------------------------------------------------------------------------------
'''Draw connections between nodes'''
# Blue color if the weight value is positive and red otherwise
# line width is based on weight value
for i, inputs in enumerate(nodes_input_):
    clr = 'indianred'
    for j, hidden in enumerate(nodes_hidden_):
        if weight1_sc[i, j] > 0:
            clr = 'steelblue'
        ax.plot([inputs[0], hidden[0]], [inputs[1], hidden[1]], lw=weight1_sc[i, j], color=clr)

for i, hidden in enumerate(nodes_hidden_):
    clr = 'indianred'
    for j, output in enumerate(nodes_output_):
        if weight2_sc[i, j] > 0:
            clr = 'steelblue'
        ax.plot([hidden[0], output[0]], [hidden[1], output[1]], lw=weight2_sc[i, j], color=clr)
# -------------------------------------------------------------------------------------------------

# Adding text to graph
layer1 = plt.text(-80, -6.5, 'Input Layer', fontweight= 'bold', fontsize =12)
layer2 = plt.text(-86, 44.5, 'Hidden Layer', fontweight='bold', fontsize = 12)
layer3 = plt.text(-76,75,'Output layer', fontweight='bold', fontsize=12)
ax._add_text(layer1)
plt.title('Connectivity Graph of 3 Layer Neural Network')

# -------------------------------------------------------------------------------

ax.set_xlim(-70, 70)  
ax.set_ylim(-10, 80)  
ax.set_aspect('equal')
ax.axis('off')

plt.show()