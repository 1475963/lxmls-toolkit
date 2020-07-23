import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, .01)

f_x = np.power(x, 2)

plt.xlim(-5, 5)
plt.ylim(-5, 30)

plt.plot(x, f_x)

points = np.array([-2, 0, 2])
plt.plot(points, np.power(points, 2), 'bo')
for point in points:
	plt.plot(x, (2 * point) * x - (np.power(point, 2)))
