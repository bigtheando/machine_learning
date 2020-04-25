import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    n_samples = 10
    n_features = 2
    noise = 30
    bias = 1
    X, Y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           noise=noise,
                           bias=bias)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    Y = (Y - Y.mean()) / Y.std()
    Y = Y.reshape(n_samples, 1)
    # 線形カーネル
    # TODO: 線形カーネル以外も実装する
    XXT = X.dot(X.T)
    # 精度行列
    XXT_inv = np.linalg.inv(XXT)

    # 推論用入力データ
    X_test = np.random.rand(2)
    # 入力データと学習データの共分散
    k_aster = X_test.dot(X.T)
    # 入力データと入力データの共分散
    k_asteraster = X_test.dot(X_test)
    # P(Y)がしたがうガウス分布の平均
    Y_mean = k_aster.dot(XXT_inv).dot(Y)
    # P(Y)がしたがうガウス分布の分散
    Y_var = k_asteraster - k_aster.T.dot(XXT_inv).dot(k_aster)
    # 予測結果
    result = np.random.normal(Y_mean[0], Y_var)
    
    """
    描画部
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    # 青点が学習データ
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
    # 橙点が推論結果
    ax.scatter(X_test[0], X_test[1], result)
    plt.show()