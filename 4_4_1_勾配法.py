# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:09:30 2018

@author: hashimoto
"""

import numpy as np
import matplotlib.pylab as plt


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


def function_2(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])  # 初期値
lr = 0.1  # 学習率
step_num = 20  # 勾配法繰り返し回数

x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

print(x_history)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
