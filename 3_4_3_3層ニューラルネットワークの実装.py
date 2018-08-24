# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 21:15:26 2018

@author: hashimoto
"""

import numpy as np


def sigmoid(x):
    """
    シグモイド関数

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


def init_network():
    """
   　ネットワーク(重み、バイアス)初期化

    Parameters
    -------
    - : -
        -

    Returns
    -------
    - : -
        -
    """
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    """
   　順伝播

    Parameters
    -------
    network : ネットワーク
    x : 入力データ(array)

    Returns
    -------
    y : 出力データ(array)
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 1層
    a1 = np.dot(x, W1) + b1
    print(a1)
    z1 = sigmoid(a1)
    print(z1)

    # 2層
    a2 = np.dot(z1, W2) + b2
    print(a2)
    z2 = sigmoid(a2)
    print(z2)

    # 3層
    a3 = np.dot(z2, W3) + b3
    print(a3)
    y = a3

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
