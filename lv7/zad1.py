import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(
            n_samples=n_samples,
            centers=4,
            cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
            random_state=random_state
        )

    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)

    # 2 grupe
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)

    else:
        X = []

    return X

X = generate_data(500, 5)  

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Podatkovni primjeri')
plt.show()

K = 3
kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=0)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Rezultat K-means grupiranja')
plt.show()

print("Vrijednost J (inertia):", kmeans.inertia_)

J = []
vrijednosti_K = range(1, 11)
for k in vrijednosti_K:
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
    model.fit(X)
    J.append(model.inertia_)

plt.figure()
plt.plot(vrijednosti_K, J, marker='o')
plt.xlabel('Broj grupa K')
plt.ylabel('J / inertia')
plt.title('Lakat metoda')
plt.show()