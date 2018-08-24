# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 01:22:24 2018

@author: hashimoto
"""

import time
import os
import sys
import numpy as np
from mnist import load_mnist
from PIL import Image
sys.path.append(os.pardir)


def img_show(img):
    """
    MNIST画像表示

    Parameters
    ----------
    img : image
        画像

    Returns
    -------
    - : -
        -
    """
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# 処理前の時刻
t1 = time.time()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)

# 処理後の時刻
t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(str(elapsed_time) + '[sec]')
