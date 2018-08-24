# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 21:15:26 2018

@author: hashimoto
"""

import matplotlib.pylab as plt


def show_2D_grahp(title, x, y, xlabel, ylabel, xmin, xmax, ymin, ymax, arg):
    """
    2次元グラフ表示

    Parameters
    ----------
    x : array
        x軸データ
    y : array
        x軸データ

    Returns
    -------
    - : -
        -
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xmin != xmax:
        plt.xlim(xmin, xmax)

    if ymin != ymax:
        plt.ylim(ymin, ymax)

    plt.grid()
    plt.plot(x, y, arg)
    plt.show()
