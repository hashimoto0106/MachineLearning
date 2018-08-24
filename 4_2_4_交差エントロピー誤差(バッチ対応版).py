# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:49:07 2018

@author: hashimoto
"""

import numpy as np


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
