import numpy as np


file = '/disk3/yangle/diagnose/code/test.txt'
data = np.random.randn(2, 5)
np.savetxt(file, data)
