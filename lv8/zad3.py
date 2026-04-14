import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image

# ucitaj spremljeni model
model = keras.models.load_model("mnist_model.keras")

# ucitaj sliku test.png
img = Image.open("C:/Users/dtros/Desktop/osu/OSU_LV/lv8/test.png").convert("L")

# promijeni velicinu na 28x28
img = img.resize((28, 28))

# pretvori u numpy polje
img_array = np.array(img)

# opcionalno prikaz slike
plt.imshow(img_array, cmap="gray")
plt.title("Ucitan test.png")
plt.axis("off")
plt.show()

# skaliranje na [0,1]
img_array = img_array.astype("float32") / 255

# ako je potrebno, invertiranje boja
# MNIST najcesce ima crnu pozadinu i svijetlu znamenku
img_array = 1 - img_array

# dodavanje dimenzija da odgovara modelu
img_array = np.expand_dims(img_array, axis=-1)   # (28, 28, 1)
img_array = np.expand_dims(img_array, axis=0)    # (1, 28, 28, 1)

# klasifikacija
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

print("Predvidena znamenka je:", predicted_class)