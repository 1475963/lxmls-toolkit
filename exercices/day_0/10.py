from lxmls.readers import galton
import matplotlib.pyplot as plt
import numpy as np

data = galton.load()

print('data:', data)
print('global mean height:', data.mean())
print('global std height:', data.std())

print('fathers mean height:', data[:, 0].mean())
print('fathers std height:', data[:, 0].std())

print('sons mean height:', data[:, 1].mean())
print('sons std height:', data[:, 1].std())


plt.figure()
plt.hist(data.ravel())

plt.figure()
plt.plot(data[:, 0], data[:, 1], '.')

data_with_noise = data + (.3 * np.random.randn(*data.shape))

plt.figure()
plt.plot(data_with_noise[:, 0], data_with_noise[:, 1], '.')
