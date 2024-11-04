import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Make perceptron class:

class Perceptron():
    # Initializing "self" instance
    def __init__(self, eta = 0.01, n_iter = 10, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc = 0.0, scale = 0.01, size = x.shape[1]+1)

        self.errors = []
        for i in range(self.n_iter):
            errors_ = 0

            for xi, target in zip(x,y): # x: [n_samples, n_features], y: [n_samples]
                update = self.eta*(target - self.predict(xi))
                self.w[1:] += update*xi
                self.w[0] += update
                errors_ += int(update != 0.0)
            
            self.errors.append(errors_)

    def net_input(self, x): # z
        return np.dot(x, self.w[1:]) + self.w[0]*1
    
    def predict(self, x): # classification
        return np.where(self.net_input(x) >= 0.0, 1, -1)



ppn = Perceptron(eta = 0.001, n_iter = 100, random_state= 2)



