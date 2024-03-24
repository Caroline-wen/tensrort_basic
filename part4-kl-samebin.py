import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
 
def cal_kl(p, q):
    KL = 0.
    for i in range(len(p)):
        KL += p[i] * np.log(p[i] / q[i])
    return KL

def KL_test(x, kl_threshold = 0.01):
    y_out = []
    while True:
        y = [np.random.uniform(1, size + 1) for i in range(size)]
        y = y / np.sum(y)
        kl_resault = cal_kl(x, y)
        if kl_resault < kl_threshold:
            plt.plot(x)
            plt.plot(y)
            break
    return y_out


if __name__ == '__main__':
    np.random.seed(1)
    size = 10
    x = [np.random.uniform(1, size + 1) for i in range(size)]
    x = x / np.sum(x)
    y_out = KL_test(x, kl_threshold = 0.01)
    plt.savefig("./kl.jpg")
    plt.show()
    print(x, y_out)
