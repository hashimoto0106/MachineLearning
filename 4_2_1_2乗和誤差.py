# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 17:48:34 2018

@author: hashimoto
"""

import numpy as np
import time


def mean_squared_error(y, t):
    """
    2乗和誤差算出

    Parameters
    ----------
    y : double
        softmax関数出力(ニューラルネットワーク出力)
    t : double
        教師データ(正解ラベル/1:正解,0:不正解)

    Returns
    -------
    fruit_price : double
        2乗和誤差
    """
    return 0.5 * np.sum((y - t)**2)

# 処理前の時刻
t1 = time.time()


t = [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

# [2](0.60)の確率が最も高い場合
y = [0.10, 0.05, 0.60, 0.00, 0.05, 0.10, 0.00, 0.10, 0.00, 0.00]
print(mean_squared_error(np.array(y), np.array(t)))

# [7](0.60)の確率が最も高い場合
y = [0.10, 0.05, 0.10, 0.00, 0.05, 0.10, 0.00, 0.60, 0.00, 0.00]
print(mean_squared_error(np.array(y), np.array(t)))

print("1つ目の例の方が損失関数の値が小さくなっているため、教師データとの誤差が小さいことが分かる。すなわち教師データと適合している")

# 処理後の時刻
t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(str(elapsed_time))
