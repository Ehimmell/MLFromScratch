import torch
from sklearn.datasets import make_regression
import classes
import helper


def linear(X = None, y = None, lr = 0.01, epochs = 100):
    X, y = helper.check_X_y(X, y)
    X, y = helper.tensor_check(X, y)
    model = classes.LinearRegression(X.shape[1], lr)
    model.fit(X, y, epochs, lr)
    return model



