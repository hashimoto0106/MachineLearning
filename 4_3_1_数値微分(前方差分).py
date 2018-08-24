# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:43:33 2018

@author: hashimoto
"""

def numerical_diff(f, x):
    """
    数値微分(前方差分)

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
    h = 10e-50
    return (f(x+h) - f(x)) / h
