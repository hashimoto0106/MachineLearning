# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:43:33 2018

@author: hashimoto
"""

import time
import os
import logging.config


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


def function_2(x, y):
    # return np.sum(x**2)
    return x[0]**2 + x[1]**2


def function_tmp1(x0):
    return x0*x0 + 4.0**2.0


def function_tmp2(x1):
    return 3.0**2.0 + x1*x1


os.system('cls')

# ログ設定ファイルからログ設定を読み込み
logging.config.fileConfig('logging.conf')
logger = logging.getLogger()

logger.info('Start')

# 処理前の時刻
t1 = time.time()

print(numerical_diff(function_tmp1, 3))
print(numerical_diff(function_tmp2, 4))

# 処理後の時刻
t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(str(elapsed_time) + '[sec]')

logger.info('End')
