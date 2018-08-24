# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:43:33 2018

@author: hashimoto
"""

import numpy as np
import time
import plot as plt


def numerical_diff(f, x):
    """
    数値微分(中心差分)

    Parameters
    ----------
    f : function
        関数
    x : double
        f(x)

    Returns
    -------
    - : double
        微分
    """
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


# 処理前の時刻
t1 = time.time()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

# 処理後の時刻
t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(str(elapsed_time) + '[sec]')

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.show_2D_grahp("Numerical Diff", x, y, "x", "f(x)", -1, -1, -1, -1, "r")
