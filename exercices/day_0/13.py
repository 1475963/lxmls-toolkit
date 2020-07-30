from lxmls.readers import galton
import matplotlib.pyplot as plt
import numpy as np

def error_func(x, y, w):
	return np.power(np.matmul(x.T, w) - y, 2)

def grad_error_func(x, y, w):
	return np.matmul(2 * x, np.matmul(x.T, w) - y) / len(y)

def gradient_descent(x, y, start_w, lr, prec):
	cur_w = start_w
	prev_w = start_w + (prec * 2)
	res = [cur_w]
	mses = []
	while abs(prev_w - cur_w).sum() > prec:
		prev_w = cur_w
		cur_w = prev_w - (lr * grad_error_func(x, y, cur_w))
		res.append(cur_w)

		error = error_func(x, y, cur_w)
		mses.append(error.mean())
	return np.array(res), np.array(mses)

data = galton.load()

print('data shape:', data.shape)

x = np.vstack([data[:, 1], np.ones(data.shape[0])])
y = data[:, 0]
w = np.array([.5, 50.0])

print('w:', w)

print('y.shape:', y.shape)
print('x.shape:', x.shape)
print('w.shape:', w.shape)

print('derivative:', grad_error_func(x, y, w))

res, mses = gradient_descent(x, y, w, 0.0002, 0.00001)

print('res:', res)
print('mses:', mses)

print('final weights w:', res[-1])

plt.plot(res[:, 0], res[:, 1], '.')
plt.show()

plt.figure()
plt.plot(mses)

print('mse with final weights:', np.power(y - np.matmul(res[-1], x), 2).mean())

m, c = np.linalg.lstsq(x.T, y)[0]

res_solver = np.array([m, c])

print('m:', m)
print('c:', c)

print('mse with solver weights:', np.power(y - np.matmul(res_solver, x), 2).mean())

plt.figure()

data_noise = data + (.3 * np.random.randn(*data.shape))
X = data_noise[:, 0]
Y = data_noise[:, 1]
X_min, X_max = [min(X), max(X)]
X_range = np.linspace(X_min, X_max, X_max - X_min + 1)
Y_pred_sgd = res[-1][0] * X_range + res[-1][1]
Y_pred_solver = res_solver[0] * X_range + res_solver[1]

plt.plot(X, Y, '.')
plt.plot(X_range, Y_pred_sgd, '--', c='r', linewidth=2)
plt.plot(X_range, Y_pred_solver, '--', c='g', linewidth=2)

