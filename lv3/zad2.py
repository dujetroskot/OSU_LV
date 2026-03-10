import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data_C02_emission.csv")
data.columns = data.columns.str.strip()

#a
plt.figure()
plt.hist(data['CO2 Emissions (g/km)'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Frequency")
plt.title("Histogram CO2 emisija")
plt.show()

# #b
plt.figure()
sns.scatterplot(
    x='Fuel Consumption City (L/100km)',
    y='CO2 Emissions (g/km)',
    hue='Fuel Type',
    data=data
)


plt.title("Odnos gradske potrošnje i CO2 emisije")
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

#c
plt.figure()
sns.boxplot(
    x='Fuel Type',
    y='Fuel Consumption Hwy (L/100km)',
    data=data
)

plt.title("Izvangradska potrošnja po tipu goriva")
plt.show()

#d
fuel_counts = data.groupby('Fuel Type').size()

fuel_counts.plot(kind='bar', color='orange')
plt.title("Broj vozila po tipu goriva")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Vehicles")
plt.show()

#e
avg_co2 = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()

avg_co2.plot(kind='bar', color='green')
plt.title("Prosječna CO2 emisija po broju cilindara")
plt.xlabel("Cylinders")
plt.ylabel("Average CO2 Emissions (g/km)")
plt.show()