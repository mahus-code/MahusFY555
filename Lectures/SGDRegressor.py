import matplotlib.pyplot as plt
import pickle

with open("z.txt", "rb") as fp:
    z = pickle.load(fp)
with open("dL.txt", "rb") as fp:
    dL = pickle.load(fp)
    
z = z.reshape(-1,1)

from sklearn.linear_model import SGDRegressor
sgd_regressor = SGDRegressor(max_iter = 1000, eta0 = 0.02, random_state = 42)

lin_r = sgd_regressor.fit(z, dL)
y_pred = lin_r.predict(z)
plt.plot(z, dL, '*')
plt.plot(z,y_pred,'-')
print(lin_r.coef_)
print(lin_r.intercept_)

from sklearn.preprocessing import PolynomialFeatures
quadratic = PolynomialFeatures(degree = 2)
z_quad = quadratic.fit_transform(z)
#print(z)
#print(z_quad)

poly_r = sgd_regressor.fit(z_quad,dL)
dL_quad_fit = poly_r.predict(z_quad)

plt.plot(z, dL_quad_fit, '--')
plt.show()

print(poly_r.coef_)
print(poly_r.intercept_)