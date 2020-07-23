import numpy as np

m = 3
n = 2
a = np.zeros([m, n])
print(a)

print(a.shape)
print(a.dtype.name)

a = np.zeros([m, n], dtype=int)
print(a.dtype)

a = np.array([[2, 3], [3, 4]])
print(a)
