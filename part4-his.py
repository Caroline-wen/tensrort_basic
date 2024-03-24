import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

data = np.random.rand(1000)
plt.hist(data, bins=200)
plt.xlabel("value")
plt.ylabel("frequency")
plt.savefig("./hist.jpg")
plt.show()