import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Define our Adaline class
class Adaline():
    def __init__(self, eta = 0.01, n_iter = 10, random_state = 1): # Takes same arguments as perceptron
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self,x): # compute z 
        return np.dot(x, self.w[1:])+self.w[0] # sum of weights times x + bias weight
    
    def activation(self, z): # adaline linear activation function
        return z

    def predict(self,x): # give class label. This is the decision function (same for adaline as for perceptron)
        return np.where(self.activation(self.net_input(x))>=0.0,1,-1) # assign 1 if true and -1 if not
    
    def fit(self, x,y): # This is the learning rule, also called "fitting", where we update the values of w
        # (x has dimensions [n_samples, n_features], y has dimension [n_sample])
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc = 0.0, scale = 0.1, size = 1+x.shape[1])
        
        self.cost_ = [] # cost function, J, sum of squared errors in this case
        for i in range(self.n_iter): # we use the entire data set per iteration
            net_input =  self.net_input(x) 
            output = self.activation(net_input)
            errors = (y - output)
            
            self.w[1:] += self.eta * x.T.dot(errors) # Update errors (= -eta * nabla J(w))
            # where nabla J(w) = sum_i (y_i - output_i) * x_i
            # x -> n x 1 and errors -> n x 1 ---> so we need x.T
            self.w[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0 # SSE
            self.cost_.append(cost)
        return self
    

''' Now we can import a dataset and fit weights. We will use the iris dataset which has 150 data points 
    representing three types of iris flowers. The first 100 data points correspond to only two types of flowers
    and we will only be using the first 100 datapoints for now. '''

#Import data from webpage:
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.head()
y=df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa', 1,-1)
x = df.iloc[0:100,[0,2]].values

# Call three instances of our adaline class (each with their own eta value)
ada_lowEta = Adaline(eta = 0.0001, n_iter = 10, random_state = 10)
ada_medEta = Adaline(eta=0.001, n_iter = 10, random_state = 10)
ada_highEta = Adaline(eta=0.01, n_iter = 10, random_state = 10)

# Teach each model using our training data
ada_lowEta.fit(x, y)
ada_medEta.fit(x, y)
ada_highEta.fit(x, y)

# We can then plot the Sum of Squared Errors (SSE) versus epochs for each model
# Shown is both a subplot, to show how the low learning rate results in the model 'converging'
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Adaline models with different learning rates')

# Low eta
ax1.plot(range(1, len(ada_lowEta.cost_)+1), np.log10(ada_lowEta.cost_), marker='o', color='coral')
ax1.grid(True)
ax1.set_ylabel(r'$\log_{10}$of SSE')
ax1.set_xlabel('Number of epochs')
ax1.set_title('Low learning rate $\eta = 0.0001$')

# Medium eta
ax2.plot(range(1, len(ada_medEta.cost_)+1), np.log10(ada_medEta.cost_), marker='o', color='slateblue')
ax2.grid(True)
ax2.set_ylabel(r'$\log_{10}$of SSE')
ax2.set_xlabel('Number of epochs')
ax2.set_title('Medium learning rate $\eta = 0.001$')

# High eta
ax3.plot(range(1, len(ada_highEta.cost_)+1), np.log10(ada_highEta.cost_), marker ='o', color='mediumvioletred')
ax3.grid(True)
ax3.set_ylabel(r'$\log_{10}$of SSE')
ax3.set_xlabel('Number of epochs')
ax3.set_title('High learning rate $\eta = 0.01$')

plt.show()

# A second plot is also done, where each eta value is depicted on the same plot
fig, ax = plt.subplots()
fig.suptitle('Adaline models with different learning rates')

ax.plot(range(1, len(ada_lowEta.cost_) + 1), np.log10(ada_lowEta.cost_), marker='o', color='coral', label='Low learning rate $\eta = 0.0001$')
ax.plot(range(1, len(ada_medEta.cost_) + 1), np.log10(ada_medEta.cost_), marker='o', color='slateblue', label='Medium learning rate $\eta = 0.001$')
ax.plot(range(1, len(ada_highEta.cost_) + 1), np.log10(ada_highEta.cost_), marker='o', color='mediumvioletred', label='High learning rate $\eta = 0.01$')

ax.grid(True)
ax.set_ylabel(r'$\log_{10}$ of SSE')
ax.set_xlabel('Number of epochs')
ax.legend()

plt.show()