import numpy as np

def iterate(x):
	n = 0
	while x <= 1:
		x = max(0, x + np.random.normal(scale=.0625))
		n += 1
	return n

iterations = np.array([iterate(0) for i in np.arange(0, 1000, 1)])

print(np.mean(iterations))
