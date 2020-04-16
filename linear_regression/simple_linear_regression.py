import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # data作成
    x_len = 50
    x, y = make_regression(random_state=0,
                           n_samples=x_len,
                           n_features=1,
                           noise=10.0,
                           bias=1.0)
    x = x[:, 0]
    x_sum = x.sum()
    x_square_sum = np.power(x, 2).sum()
    y_sum = y.sum()
    xy_sum = (x * y).sum()
    mat = np.array([[x_len, x_sum],
                    [x_sum, x_square_sum]])
    vec = np.array([[y_sum],
                    [xy_sum]])
    a, b = np.linalg.inv(mat).dot(vec)
    a = a[0]
    b = b[0]
    x_min = x.min()
    x_max = x.max()
    x_hat = np.arange(x_min - 1, x_max + 1)
    y_hat = a + b * x_hat

    plt.scatter(x, y, color="b")
    plt.show()

    plt.scatter(x, y, color="b")
    plt.plot(x_hat, y_hat, color="r")
    plt.show()