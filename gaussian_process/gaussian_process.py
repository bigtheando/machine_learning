import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_samples = 10
    n_features = 1
    noise = 30
    bias = 1
    X, Y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           noise=noise,
                           bias=bias)
                           
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    Y = Y.reshape(n_samples, 1)
    XXT = X.dot(X.T)

    X_test = np.random.rand()
    X_test = (X_test - X_mean) / X_std
    X_test = np.array([X_test])
    k_aster = X_test.dot(X.T)
    Y_hat = k_aster.dot(XXT).dot(Y)

    plt.scatter(X, Y)
    plt.scatter(X_test, Y_hat)
    plt.show()