# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 18:58:08 2018

@author: hashimoto
"""

import numpy as np
import time
import matplotlib.pylab as plt


def cross_entropy_error(y, t):
    """
    交差エントロピー誤差

    Parameters
    ----------
    y : double
        softmax関数出力(ニューラルネットワーク出力)
    t : double
        教師データ(正解ラベル/1:正解,0:不正解)

    Returns
    -------
    - : double
        交差エントロピー誤差
    """
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 処理前の時刻
t1 = time.time()


t = [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

# [2](0.60)の確率が最も高い場合
y = [0.10, 0.05, 0.60, 0.00, 0.05, 0.10, 0.00, 0.10, 0.00, 0.00]
print(cross_entropy_error(np.array(y), np.array(t)))

# [7](0.60)の確率が最も高い場合
y = [0.10, 0.05, 0.10, 0.00, 0.05, 0.10, 0.00, 0.60, 0.00, 0.00]
print(cross_entropy_error(np.array(y), np.array(t)))

print("yが小さければ損失関数の値は大きくなる")

# 処理後の時刻
t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(str(elapsed_time))


x = np.arange(0.0, 1.0, 0.1)
y = np.log(x)
plt.plot(x, y)
plt.ylim(-5.0, -0.1)
plt.show()
