import threading
import numpy as np
import time

start = time.perf_counter()

def compute():
    for i in range(1, 1000):
        a = np.random.rand(i)
        print(a)
        sum = 0
        sum += a[0]
    print(sum)


if __name__ == '__main__':

    t = threading.Thread(target=compute)
    t.start()
    finish = time.perf_counter()
    print('running time is ', finish-start)