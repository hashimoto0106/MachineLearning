# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 00:58:39 2018

@author: hashimoto
"""
import time
import os
import sys
import numpy as np
import pickle
from mnist import load_mnist
sys.path.append(os.pardir)


def sigmoid(x):
    """
    シグモイド関数

    Parameters
    ----------
    x : array
        入力データ

    Returns
    -------
    - : -
        出力データ
    """
    return 1 / (1+np.exp(-x))


def softmax(a):
    """
    ソフトマックス関数

    Parameters
    ----------
    a : 入力データ(array)

    Returns
    -------
    y : 出力データ(array)
    """
    c = np.max(a)
    exp_a = np.exp(a - c)     # オーバフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def get_data():
    """
   　ネットワーク(重み、バイアス)初期化

    Parameters
    -------
    - : -
        -

    Returns
    -------
    x_test : 画像データ
    t_test : 正解ラベル
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """
   　ネットワーク(重み、バイアス)初期化

    Parameters
    -------
    - : -
        -

    Returns
    -------
    - : -
        -
    """
    # 学習済パラメータ（重みとバイアスがディクショナリ型で保存）読み込み
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    """
   　予測

    Parameters
    -------
    network : ネットワーク
    x : 入力画像データ(array)

    Returns
    -------
    y : 確率データ
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# 処理前の時刻
t1 = time.time()

# 入力層:784個(画像サイズ28×28),出力層：１０個(数字0-9)
x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    # 最も確率の高い要素のインデックスを取得
    p = np.argmax(y)
    
    # 予測データと正解ラベルの比較
    if p == t[i]:
        # 正解回数
        accuracy_cnt += 1

# 認識精度(正解割合)
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 処理後の時刻
t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(str(elapsed_time) + '[sec]')
