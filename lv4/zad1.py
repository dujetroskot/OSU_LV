import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Učitavanje podataka
df = pd.read_csv("data_C02_emission.csv")

# a) Odabir stupaca i podjela 80/20
input_columns = ["Engine Size (L)", 
                 "Cylinders", 
                 #"Fuel Consumption Comb (L/100km)", 
                 #"Fuel Consumption City (L/100km)", 
                 #"Fuel Consumption Hwy (L/100km)"
                 ]
output_column = "CO2 Emissions (g/km)"

X = df[input_columns].values
y = df[output_column].values
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# b) Dijagram raspršenja
plt.figure()
for i in range(len(input_columns)):
    plt.scatter(X_train[:, i], y_train, color="blue", alpha=0.4, label="Učenje")
    plt.scatter(X_test[:, i],  y_test,  color="red",  alpha=0.4, label="Test")
    plt.xlabel(input_columns[0])
    plt.ylabel(output_column)
    plt.legend()
    plt.title("CO2 emisija vs ...")
    plt.show()

# c) Standardizacija - fit samo na train, transform na oba
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Histogram jedne veličine prije i nakon
plt.figure()
plt.subplot(1, 2, 1)
plt.hist(X_train[:, 0], bins=30, color="blue")
plt.title("Prije skaliranja")

plt.subplot(1, 2, 2)
plt.hist(X_train_sc[:, 0], bins=30, color="orange")
plt.title("Nakon skaliranja")
plt.show()

# d) Izgradnja modela i ispis parametara
model = LinearRegression()
model.fit(X_train_sc, y_train)

print("Parametri modela:")
print(f"  theta_0 (slobodni clan) = {model.intercept_:.4f}")
for i, (col, coef) in enumerate(zip(input_columns, model.coef_)):
    print(f"  theta_{i+1} ({col}) = {coef:.4f}")

# e) Procjena i dijagram stvarno vs procijenjeno
y_pred = model.predict(X_test_sc)

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Procijenjene vrijednosti")
plt.title("Stvarno vs Procijenjeno")
plt.show()

# f) Metrike
print("\nMetrike na testnom skupu:")
print(f"  MAE  = {mean_absolute_error(y_test, y_pred):.4f}")
print(f"  MSE  = {mean_squared_error(y_test, y_pred):.4f}")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R2   = {r2_score(y_test, y_pred):.4f}")
print(f"  MAPE   = {mean_absolute_percentage_error(y_test, y_pred):.4f}")