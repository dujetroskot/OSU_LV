import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("C:/Users/dtros/Desktop/osu/OSU_LV/lv7/imgs/imgs/test_3.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transformiraj sliku u 2D numpy polje
w, h, d = img.shape
img_array = np.reshape(img, (w * h, d))

# broj razlicitih boja u originalnoj slici
broj_boja = len(np.unique((img_array * 255).astype(np.uint8), axis=0))
print("Broj razlicitih boja u originalnoj slici:", broj_boja)

# odabir broja grupa
K = 3

# primjena K-means algoritma
kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=0)
kmeans.fit(img_array)

# oznaka klastera za svaki piksel
labels = kmeans.labels_

# centri klastera = dominantne boje
centers = kmeans.cluster_centers_

# svaki piksel zamijeni pripadnim centrom
img_array_aprox = centers[labels]

# vrati natrag u oblik slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

# prikazi kvantiziranu sliku
plt.figure()
plt.title(f"Kvantizirana slika, K={K}")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

broj_boja_nova = len(np.unique((img_array_aprox * 255).astype(np.uint8), axis=0))
print("Broj razlicitih boja nakon kvantizacije:", broj_boja_nova)

print("Vrijednost J (inertia):", kmeans.inertia_)

J = []
vrijednosti_K = range(1, 11)

for k in vrijednosti_K:
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
    model.fit(img_array)
    J.append(model.inertia_)

plt.figure()
plt.plot(vrijednosti_K, J, marker='o')
plt.xlabel("Broj grupa K")
plt.ylabel("J / inertia")
plt.title("Lakat metoda")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))

for i in range(K):
    maska = (labels == i).astype(np.uint8)
    maska = np.reshape(maska, (w, h))

    plt.subplot(1, K, i + 1)
    plt.title(f"Klaster {i}")
    plt.imshow(maska, cmap='gray')
    plt.axis("off")

plt.tight_layout()
plt.show()