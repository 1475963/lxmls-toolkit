import numpy as np

a = np.array([[2, 3], [3, 4]])
b = np.array([[1, 1], [1, 1]])

a_dim1, a_dim2 = a.shape
b_dim1, b_dim2 = b.shape
c = np.zeros((a_dim1, b_dim2))
for i in range(a_dim1):
	for j in range(b_dim2):
		for k in range(a_dim2):
			c[i, j] += a[i, k] * b[k, j]

print(a * b)
print(c)
print(np.dot(a, b))
