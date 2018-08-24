# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:58:27 2018

@author: hashimoto
"""

import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


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


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
