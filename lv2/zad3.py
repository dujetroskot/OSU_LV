import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("C:/Users/student/Desktop/lv2/OSU_LV/lv2/road.jpg")

bright = img + 50
bright = np.clip(bright, 0, 255)
plt.imshow(bright)
plt.title("Posvijetljena slika")
plt.show()


h, w, c = img.shape
druga_cetvrtina = img[:, w//4:w//2, :]
plt.imshow(druga_cetvrtina)
plt.title("Druga četvrtina po širini")
plt.show()


rotirana = np.rot90(img, k=-1)
plt.imshow(rotirana)
plt.title("Rotirana 90°")
plt.show()


zrcaljena = np.fliplr(img)
plt.imshow(zrcaljena)
plt.title("Zrcaljena slika")
plt.show()