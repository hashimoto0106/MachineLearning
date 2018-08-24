# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:20:23 2018

@author: hashimoto
"""

import numpy as np


def softmax(a):
    """
    ソフトマックス関数

    Parameters
    ----------
    a : 入力データ(array)

    Returns
    -------
    y : 出力データ(array)
    """
    c = np.max(a)
    exp_a = np.exp(a - c)     # オーバフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """
    交差エントロピー誤差(バッチ対応版)

    Parameters
    ----------
    y : softmax関数出力(ニューラルネットワーク出力)
    t : 教師データ(正解ラベル/1:正解,0:不正解)

    Returns
    -------
    - : 交差エントロピー誤差(正規化)
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batchsize = y.shape[0]
    return -np.sum(np.log(y[np.arange(batchsize), t])) / batchsize


def f(W):
    return net.loss(f, net.W)


def _numerical_gradient_no_batch(f, x):
    """
    勾配法

    Parameters
    ----------
    f : 最適化したい関数
    X : 入力データ

    Returns
    -------
    grad : 勾配
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 値を元に戻す

    return grad


def numerical_gradient(f, X):
    """
    勾配法

    Parameters
    ----------
    f : 最適化したい関数
    X : 入力データ

    Returns
    -------
    grad : 勾配
    """
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    勾配降下法

    Parameters
    ----------
    f : 最適化したい関数
    init_x : 初期値
    lr : 学習率
    step_num : 勾配法繰り返し回数

    Returns
    -------
    x : 最小値
    x_history : 最小値履歴
    """
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

        # print(x)

    return x, np.array(x_history)


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


net = simpleNet()
print(net.W)  # 重みパラメータ

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))

t = np.array([0, 0, 1])
net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
