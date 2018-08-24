# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 19:31:32 2018

@author: hashimoto
"""

import numpy as np


A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))

C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[1, 2], [3, 4], [4, 5]])
print(np.dot(C, D))

print(np.dot(A, D))
