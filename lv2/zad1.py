import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 3])
y = np.array([1, 2, 2, 1])
x = np.append(x, x[0])
y = np.append(y, y[0])

plt.plot(x, y, 
         color='purple', 
         marker='o', 
         linestyle='-', 
         linewidth=3)
plt.xlabel("X os")
plt.ylabel("Y os")
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.title("Primjer")
plt.grid()

plt.show()