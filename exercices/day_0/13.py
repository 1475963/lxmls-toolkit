from lxmls.readers import galton
import numpy as np

def error_func(x, y, w):
	return np.sum([
		np.power((x_i.T * w) - y_i, 2)
		for x_i, y_i in zip(x, y)
	])

def grad_error_func(x, y, w):
	return np.sum([
		2 * x_i * (x_i.T * w - y_i)
		for x_i, y_i in zip(x, y)
	])
