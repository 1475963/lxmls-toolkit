import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1000)

Y = X ** 2

plt.plot(X, Y, 'r')

ints = np.arange(-4,5)

plt.plot(ints, ints ** 2, 'bo')

plt.xlim(-4.5, 4.5)
plt.ylim(-1, 17)
plt.show()
