import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# ucitaj spremljeni model
model = keras.models.load_model("mnist_model.keras")

# ucitaj MNIST test podatke
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# priprema slika isto kao u 1. zadatku
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

# predikcije modela
predictions = model.predict(x_test_s)
y_pred = np.argmax(predictions, axis=1)

# pronadi krivo klasificirane slike
wrong = np.where(y_pred != y_test)[0]

print("Broj krivo klasificiranih slika:", len(wrong))

# prikazi nekoliko krivo klasificiranih slika
for i in range(5):
    idx = wrong[i]
    plt.figure()
    plt.imshow(x_test[idx], cmap="gray")
    plt.title("Stvarna: " + str(y_test[idx]) + ", Predvidena: " + str(y_pred[idx]))
    plt.axis("off")
    plt.show()