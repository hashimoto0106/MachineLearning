# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 08:51:16 2018

@author: hashimoto
"""

import numpy as np


X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])

Y = np.dot(X, W)
print(Y)
