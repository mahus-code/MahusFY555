import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# -------------------------------------------------------------------------------------
# Step 1) Modifying our adaline class to do regression
# -------------------------------------------------------------------------------------

class LinearRegressionGD():
    def __init__(self, eta = 0.01, n_iter = 10, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state 

    def net_input(self,x): # compute z 
        return np.dot(x, self.w[1:])+self.w[0] # sum of weights times x + bias weight
    
    def activation(self, z): # adaline linear activation function
        return z

    def predict(self,x): # give class label. This is the decision function (same for adaline as for perceptron)
        return self.net_input(x) # we now want continous target values and not -1 or 1 
    
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

    def coef(self):
        return self.w[1:]
    
    def intercept(self):
        return self.w[0]
        
# -------------------------------------------------------------------------------------
# Step 2) Loading the Hubble data set using pickle
# -------------------------------------------------------------------------------------

with open(r"C:\Users\magnu\VsProjects\fy555_python\MahusFY555\Lectures\z.txt", "rb") as file: # read, binary
    z = pickle.load(file)
with open(r"C:\Users\magnu\VsProjects\fy555_python\MahusFY555\Lectures\dL.txt", "rb") as file:
    dL = pickle.load(file)


z = z.reshape(-1,1)
zTrain, zTest, dLTrain, dLTest = train_test_split(z,dL, test_size=0.3, stratify=None, random_state=1) # Stratification not neccessary since we
# do need to classify the data into classes

# z has shape (100,) we want it to be a column vector: (100, 1)
# We start by preprocessing our data (following the example from the book)
from sklearn.preprocessing import StandardScaler

sc_z = StandardScaler()
sc_dL = StandardScaler()

# We standardize our 4 data groups
zTrain_std = sc_z.fit_transform(zTrain)
dLTrain_std = sc_dL.fit_transform(dLTrain[:, np.newaxis]).flatten() # We add a temporary axis such that it can be transformed
zTest_std = sc_z.fit_transform(zTest)
dLTest_std = sc_dL.fit_transform(dLTest[:, np.newaxis]).flatten()


# -------------------------------------------------------------------------------------
# Step 3) Train our Linear Regression model:
# -------------------------------------------------------------------------------------

linearGD = LinearRegressionGD(eta=0.001, n_iter=100, random_state=1)

# We then fit our training data to dL = (c/H0) * z
# Note we fit based on the standardized data
lin_r = linearGD.fit(zTrain_std, dLTrain_std)

# Prediction is also made based on standardized data
dLpred_std = lin_r.predict(zTest_std)

print("Estimated slope:", lin_r.coef())
print("Estimated intercept", lin_r.intercept())

# -------------------------------------------------------------------------------------
# Step 4) Plot cost function as a function of the epochs
# -------------------------------------------------------------------------------------

import matplotlib.ticker as ticker
x = range(1, len(lin_r.cost_) + 1)
y = lin_r.cost_
plt.plot(x[::2], y[::2], marker='o', color='royalblue')
plt.grid(True)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
plt.title('Sum of squared errors vs. number of epochs of the Linear Regression model')
plt.ylabel('Sum of Squared Errors')
plt.xlabel('Epochs')

plt.show()

# -------------------------------------------------------------------------------------
# Step 5) Print r^2 value to screen
# -------------------------------------------------------------------------------------

print('Goodness of fit value (r^2) =', r2_score(dLTest_std, dLpred_std))

# Inverse transform to get the proper scale of the data again
dL_pred = sc_dL.inverse_transform(dLpred_std[:, np.newaxis]).flatten()


# -------------------------------------------------------------------------------------
# Step 6) Plot dL(z) data and our predicted model
# -------------------------------------------------------------------------------------

# We then plot the predicted dL versus z-test in their original scale
plt.plot(zTrain, dLTrain, 'o', label='Training Data', color = 'firebrick')
plt.plot(zTest, dLTest, '*', label='Test Data', color='salmon')

plt.plot(zTest, dL_pred, '-', label='Prediction', color='dodgerblue')
# We can compare with the dlTest
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('z')
plt.ylabel('dL')
plt.title('dL vs. z with the predicted regression from \n Linear Regression model (with testing and training data visualized)')

plt.show()

# -------------------------------------------------------------------------------------
# Step 7) Using our fit and slope value --> determine H_0 
# -------------------------------------------------------------------------------------

slope_std = lin_r.coef()
slope_original = (slope_std * (sc_dL.scale_ / sc_z.scale_))  # Adjust for scaling
slope_original = slope_original[0]

# We can then use our slope value to find H0
c = 299792.458 # Given in km/s
H_0 = c/(slope_original) # in km/s/Mpc
print("H_0 (from own Linear Regression model) =", H_0, "km/s/Mpc")


# -------------------------------------------------------------------------------------
# Step 8) Using our fits from class and comparing 
# -------------------------------------------------------------------------------------

''' Using the SGD_Regressor from class '''
from sklearn.linear_model import SGDRegressor
sgd_regressor = SGDRegressor(max_iter = 1000, eta0 = 0.02, random_state = 42)

lin_r_class = sgd_regressor.fit(z, dL)
y_pred = lin_r_class.predict(z)
class_slope_lin = lin_r_class.coef_

from sklearn.preprocessing import PolynomialFeatures
quadratic = PolynomialFeatures(degree = 2)
z_quad = quadratic.fit_transform(z)

poly_r = sgd_regressor.fit(z_quad,dL)
dL_quad_fit = poly_r.predict(z_quad)

# First we print the H0 values for the models from class (as well as our own)
slope_poly = poly_r.coef_[1]

print("H_0 (from sklearn model) =", c/class_slope_lin[0], "km/s/Mpc")
print("H_0 (from polynomial model) =", c/slope_poly, "km/s/Mpc")
print("H_0 (from own Linear Regression model) =", H_0, "km/s/Mpc")

# We then plot the data plus the three different models:
plt.plot(zTrain, dLTrain, 'o', label='Training Data', color = 'firebrick', markersize=4)
plt.plot(zTest, dLTest, '*', label='Test Data', color='salmon', markersize=4)

plt.plot(zTest, dL_pred, '-', label='Prediction', color='dodgerblue')
plt.plot(z, y_pred, '--', label='Sklearn model (from class)', color='darkmagenta', linewidth = 3)
plt.plot(z, dL_quad_fit, ':', label='Polynomial fit', color='forestgreen', linewidth=3)

plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('z')
plt.ylabel('dL')
plt.title('dL vs. z with the predicted regression from \n Linear Regression model, sklearn regression model \n polynomial model (with testing and training data visualized)')

plt.show()

# Finally we print the r^2 values for all three:
print(f"{'Sklearn LinearRegression model r^2':<40} = {r2_score(dL, y_pred):.4f}")
print(f"{'Polynomial model r^2':<40} = {r2_score(dL, dL_quad_fit):.4f}")
print(f"{'Own Linear Regression model r^2':<40} = {r2_score(dLTest_std, dLpred_std):.4f}")