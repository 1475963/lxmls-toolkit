import numpy as np

i = np.linspace(0, 999, 1000)

integral = (((i / 1000) ** 2) / len(i)).sum()
print('integral:', integral)
