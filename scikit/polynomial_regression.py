import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# generate training set
X = np.array([[3, 0], [0, 3], [1, 1], [2, 3], [4, 1], [9, 6], [5, 6], [3, 1]])
poly = PolynomialFeatures(degree=2) # set degree to 2
X_trans = poly.fit_transform(X)   # transform from [x1, x2] to [1, x1, x2, x1x2, x1^2, x2^2]
coeffi = np.array([[2], [1], [-3], [2], [-5], [6]])   # parameters for y = 2 + x1 - 3 * x2 + 2 * x1 * x2 - 5 * x1^2 + 6 * x2^2
y = np.dot(X_trans, coeffi)
random_matrix = np.random.random((len(y), 1)) - 0.5
y = y + random_matrix   # make y randomly

# solve as normal multivariate linear regression
regr = linear_model.LinearRegression()
regr.fit(X_trans, y)
print regr.coef_    # slope
print regr.intercept_   # intercept