import numpy as np
import matplotlib.pyplot as plt

dim = 50
crni = np.zeros((dim, dim, 3))
bijeli = np.ones((dim, dim, 3)) * [1,0,0] 

gornji_red = np.hstack((crni, bijeli))
donji_red = np.hstack((bijeli, crni))
slika = np.vstack((gornji_red, donji_red))

plt.imshow(slika)
plt.grid()
plt.show()