import numpy as np
from sklearn import datasets, linear_model

# z = 3 + 2 * x1 + x2
X = np.array([
    [3, 0],
    [0, 3],
    [1, 1],
    [2, 3],
    [4, 1]
])
z = np.array([
    [11 + 0.5],
    [5 + 0.7],
    [6 - 0.3],
    [11 + 0.2],
    [15 - 0.8]
])

# Plot outputs
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, z)

print regr.coef_    # slope
print regr.intercept_   # interceptor
