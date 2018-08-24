# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 09:17:37 2018

@author: hashimoto
"""

import numpy as np


def sigmoid(x):
    """
    シグモイド関数(活性化関数)

    Parameters
    ----------
    x : array
        入力データ

    Returns
    -------
    - : -
        出力データ
    """
    return 1 / (1+np.exp(-x))


X = np.array([1.0, 0.5])  # 入力
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 重み
B1 = np.array([0.1, 0.3, 0.5])  # バイアス

A1 = np.dot(X, W1) + B1
print(A1)

Z1 = sigmoid(A1)  # 活性化関数
print(Z1)
