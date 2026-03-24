import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

df = pd.read_csv("data_C02_emission.csv")

input_columns = ["Engine Size (L)", "Cylinders", "Fuel Consumption Comb (L/100km)", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Type"]
output_column = "CO2 Emissions (g/km)"
df_encoded = pd.get_dummies(df[input_columns], columns=["Fuel Type"])

X = df_encoded.values
y = df["CO2 Emissions (g/km)"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
errors = np.abs(y_test - y_pred)
max_error = np.max(errors)

print("Maksimalna pogreška:", max_error)

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Procijenjene vrijednosti")
plt.title("Stvarno vs Procijenjeno")
plt.show()

print("\nMetrike na testnom skupu:")
print(f"  MAE  = {mean_absolute_error(y_test, y_pred):.4f}")
print(f"  MSE  = {mean_squared_error(y_test, y_pred):.4f}")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R2   = {r2_score(y_test, y_pred):.4f}")
print(f"  MAPE   = {mean_absolute_percentage_error(y_test, y_pred):.4f}")
