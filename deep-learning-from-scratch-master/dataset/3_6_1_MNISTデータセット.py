import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist
import time


# 処理前の時刻
t1 = time.time()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

# 処理後の時刻
t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(str(elapsed_time) + '[sec]')
