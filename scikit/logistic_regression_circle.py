import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# Area A: x^2 + y^2 <= 2
# Area B: x^2 + y^2 > 2
X = np.array([
    # training set for A
    [0, 0],
    [1, 0],
    [1.5, -0.2],
    [1, 1.5],
    [-1, -1.5],
    [-1.9, 0],
    [-0.5, -1],
    [0, 1.9],
    [0, -1.9],
    [-1.9, 0],
    [-1.5, 0.5],
    # training set for B
    [2, 2.5],
    [2, 3],
    [1, 5],
    [1, -5],
    [-3, 0],
    [-2, -0.1],
    [-2, 1],
    [0, -2.1],
    [0, 2.1],
    [-2.1, 0],
    [-0.6, -2]
])

poly = PolynomialFeatures(degree=2)
X_trans = poly.fit_transform(X)

Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

logreg = linear_model.LogisticRegression(C=1e6)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_trans, Y)

print logreg.coef_
print logreg.intercept_
print logreg.predict(np.array(poly.fit_transform([[0, 1.8]])))  # expect to return 0
print logreg.predict(np.array(poly.fit_transform([[0, -1.8]])))  # expect to return 0
print logreg.predict(np.array(poly.fit_transform([[0, -2.1]])))  # expect to return 1