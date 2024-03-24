import numpy as np

def smooth_data(p, eps = 0.0001):
    is_zeros = (p==0).astype(np.float32) # [0. 1. 0. 0. 0. 0. 0. 0.]
    is_nonzeros = (p!=0).astype(np.float32) # [1. 0. 1. 1. 1. 1. 1. 1.]
    n_zeros = is_zeros.sum() # 7
    n_nonzeros = is_nonzeros.sum() # 1

    eps1 = (eps * n_zeros) / n_nonzeros # 1.4285714e-05: 将在0位置加上的eps平摊到非0位去减, 保证总概率为1
    hist = p.astype(np.float32)
    # print(eps * is_zeros) # [0. 1.e-04 0. 0. 0. 0. 0. 0.]
    # print(-eps1 * is_nonzeros) # [-1.4285714e-05 0 -1.4285714e-05 -1.4285714e-05 -1.4285714e-05 -1.4285714e-05 -1.4285714e-05 -1.4285714e-05]
    hist  +=  (eps * is_zeros) + ((-eps1) * is_nonzeros)
    return hist



def cal_kl(p, q):
    KL = 0.
    for i in range(len(p)):
        KL += p[i] * np.log(p[i] / q[i]) # q[i]不能为0, 可做数据平滑消除影响
    return KL



if __name__ == '__main__':
    p = [1, 0, 2, 3, 5, 3, 1, 7]
    bin = 4
    split_p = np.array_split(p, bin) # [array([1, 0]), array([2, 3]), array([5, 3]), array([1, 7])]
    q = []
    for arr in split_p:
        avg = np.sum(arr) / np.count_nonzero(arr)
        for item in arr:
            if item != 0:
                q.append(avg)
                continue
            q.append(0)
    print(q) # [1.0, 0, 2.5, 2.5, 4.0, 4.0, 4.0, 4.0]
    p /= np.sum(p)
    q /= np.sum(q)
    print("p = ", p)
    print("q = ", q)
    p = smooth_data(p)
    q = smooth_data(q)
    print("smooth_p = ", p)
    print("smooth_q = ", q)
    kl_resault = cal_kl(p, q)
    print("kl_resault = ", kl_resault)
