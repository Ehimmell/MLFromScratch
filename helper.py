import torch
from sklearn.datasets import make_regression


def tensor_check(X, y):
    if not torch.is_tensor(X):
        X = torch.tensor(X)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    return X, y


def generate_data(n_samples, n_features, noise):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
    return torch.tensor(X), torch.tensor(y)


def check_X_y(X, y):
    if (X is None) ^ (y is None):
        raise ValueError('X cannot be None while y is not None, and vice versa. Provide data for or clear both.')
    elif X is None and y is None:
        X, y = generate_data(100, 1, 0.1)
    return X, y