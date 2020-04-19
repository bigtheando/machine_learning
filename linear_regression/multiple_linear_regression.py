import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    # data作成
    X_len = 50
    X, y = make_regression(random_state=0,
                           n_samples=X_len,
                           n_features=2,
                           noise=10.0,
                           bias=1.0)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    y = y.reshape(X_len, 1)
    A = X.T.dot(X)
    w = np.linalg.inv(A).dot(X.T).dot(y)
    
    x_hat = np.arange(-5, 5).reshape(10, 1)
    y_hat = np.arange(-5, 5).reshape(10, 1)
    z_hat = np.hstack((x_hat, y_hat)).dot(w)
    x_hat = x_hat.flatten()
    y_hat = y_hat.flatten()
    z_hat = z_hat.flatten()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(np.ravel(x_hat), np.ravel(y_hat), np.ravel(z_hat), alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], zs=y)
    plt.show()