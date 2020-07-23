import numpy as np
import matplotlib.pyplot as plt

def get_y(x):
	return np.power(x + 2, 2) - (16 * np.exp(-np.power(x - 2, 2)))

def get_grad(x):
	return ((2 * x) + 4) - (16 * ((-2 * x) + 4) * np.exp(-np.power(x - 2, 2)))

def gradient_descent(start_x, func, grad, step_size=.1, prec=.0001):
	max_iter = 100
	x_new = start_x
	res = []
	for i in range(max_iter):
		x_old = x_new
		x_new = x_old - step_size * grad(x_new)
		f_x_new = func(x_new)
		f_x_old = func(x_old)
		res.append([x_new, f_x_new])

		if (abs(f_x_new - f_x_old) < prec):
			print('change in function values too small, leaving')
			return np.array(res)
	print('exceeded maximum number of iterations, leaving')
	return np.array(res)

x = np.arange(-8, 8, .001)
y = get_y(x)
plt.plot(x, y)

x_0 = 8
res = gradient_descent(x_0, get_y, get_grad, step_size=.1)
plt.plot(res[:, 0], res[:, 1], 'r+')
