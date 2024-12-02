import matplotlib.pyplot as plt
import pickle

# If you don't have data yyet:
z = [0, 0.01, 0.02]
dL = [10, 20, 30]

with open(r"C:\Users\magnu\VsProjects\fy555_python\MahusFY555\Lectures\z.txt", "rb") as fp:
    z = pickle.load(fp)
with open(r"C:\Users\magnu\VsProjects\fy555_python\MahusFY555\Lectures\dL.txt", "rb") as fp:
    dL = pickle.load(fp)

z = z.reshape(-1,1)

from sklearn.linear_model import SGDRegressor

sgd_regressor = SGDRegressor(max_iter = 1000, eta0 = 0.1, random_state = 42)

lin_r = sgd_regressor.fit(z, dL)
ypred = lin_r.predict(z)
print(lin_r.coef_)
print(lin_r.intercept_)

plt.plot(z,dL,'*')
plt.plot(z, ypred, '-')

from sklearn.preprocessing import PolynomialFeatures

quadratic = PolynomialFeatures(degree=2)
z_quad = quadratic.fit_transform(z)
#print(z_quad)

poly_r = sgd_regressor.fit(z_quad, dL)
dL_quad = poly_r.predict(z_quad)
print(poly_r.coef_)
plt.plot(z,dL_quad,'--')
plt.show()


