import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MultipleLinearRegression:
    def __init__(self, X, y, l2_coeff=1E-3):
        assert len(X) == len(y), "len(X) is not equal to len(y)"
        self.X = X
        self.y = y
        self.W = None
        # 正則化項
        self.l2_coeff = l2_coeff
    
    def fit(self):
        A = self.X.T.dot(self.X)
        self.W = np.linalg.inv(A + self.l2_coeff * np.identity(len(A))).dot(self.X.T).dot(self.y)
    
    def predict(self, X):
        return X.dot(self.W)


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

    model = MultipleLinearRegression(X=X, y=y, l2_coeff=1E-3)
    model.fit()

    # 推論    
    X_hat = np.random.rand(10, 2)
    y_hat = model.predict(X=X_hat)

    """
    描画部
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    # 青点が学習用データ
    ax.scatter(X[:, 0], X[:, 1], zs=y)
    # 橙点が推論結果
    ax.scatter(X_hat[:, 0], X_hat[:, 1], y_hat)
    plt.show()