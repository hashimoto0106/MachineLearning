# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 23:47:48 2018

@author: hashimoto
"""

import time
import os
import sys
import numpy as np
from mnist import load_mnist
sys.path.append(os.pardir)

# 処理前の時刻
t1 = time.time()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10

# 0-60000からランダムに10個の数値を選ぶ
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 処理後の時刻
t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(str(elapsed_time) + '[sec]')

print(batch_mask)
# print(x_batch)
# print(t_batch)
