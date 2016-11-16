import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# data
x, y, coef = make_regression(
    n_samples=200,
    n_features=3,
    n_informative=2,
    random_state=1,
    noise=4.0,
    bias=100.0,
    coef=True
    )

print('Data generated. Underlying model is Y = X{0} + c + e'.format(coef))

# Create linear regression object
linear_regr = LinearRegression()

# Train the model using the training sets
linear_regr.fit(x, y)
mse = np.mean((linear_regr.predict(x) - y) ** 2)

# The coefficients
print('Coefficients: \n', linear_regr.coef_)
# The mean squared error
print("Mean squared error: {:.2f}".format(mse))
# Explained variance score: 1 is perfect prediction
print('Variance score: {:.2f}'.format(linear_regr.score(x, y)))
