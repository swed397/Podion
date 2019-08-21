import numpy as np
import matplotlib.pyplot as plt
import time

a = np.random.randn(10, 10)
a[4, :] = 5
a[:, 4] = 5
plt.matshow(a)
plt.show()
t0 = time.perf_counter()
a[a == 5] = 10
print(time.perf_counter() - t0, a.shape)
