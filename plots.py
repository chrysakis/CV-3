import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3 * np.pi, 1000)
y = np.sin(10 * np.sin(x))
z = np.sin(x**2)

fig = plt.figure()
plt.subplot(211)
plt.plot(x, y)
plt.subplot(212)
plt.plot(x, z)
plt.show()
fig.savefig('../plots/subplot.png', bbox_inches='tight')
