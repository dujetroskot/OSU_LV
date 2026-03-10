import pandas as pd
import numpy as np

data = pd.read_csv('C:/Users/student/Desktop/lv3/OSU_LV/lv3/data_C02_emission.csv')

#a
print(f"Broj mjerenja = {len(data)}")
print(f"Tipovi varijabli = {data.dtypes}")
print(f"Broj mjerenja = {data.isnull().sum()}")
print(f"Broj mjerenja = {data.duplicated().sum()}")
data = data.dropna()
data = data.drop_duplicates()
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category')

print("\nTipovi nakon konverzije:")
print(data.dtypes)

#b
print("\n3 vozila s NAJVEĆOM gradskom potrošnjom:")
print(data.nlargest(3, 'Fuel Consumption City (L/100km)')[['Make','Model','Fuel Consumption City (L/100km)']])
print("\n3 vozila s NAJMANJOM gradskom potrošnjom:")
print(data.nsmallest(3, 'Fuel Consumption City (L/100km)')[['Make','Model','Fuel Consumption City (L/100km)']])

#c
filtered = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]

print("\nBroj vozila (2.5–3.5 L):", filtered.shape[0])
print("Prosječna CO2 emisija:", filtered['CO2 Emissions (g/km)'].mean())

#d
audi = data[data['Make'] == 'Audi']
print(f'Broj audi vozila: {data.shape[0]}')
print(data.columns)
audi4cylinders = audi[audi['Cylinders'] == '4']
print(f'Prosjcna emisija: {data['CO2 Emissions (g/km)'].mean()}')

#e
print("\nBroj vozila po cilindrima:")
print(data['Cylinders'].value_counts().sort_index())

print("\nProsječna CO2 emisija po cilindrima:")
print(data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean())

#f
diesel = data[data['Fuel Type'] == 'Z']
gas = data[data['Fuel Type'] == 'X'] 

print("\nProsječna gradska potrošnja (diesel):",
    diesel['Fuel Consumption City (L/100km)'].mean())
print("Medijan (diesel):",
    diesel['Fuel Consumption City (L/100km)'].median())

print("\nProsječna gradska potrošnja (regular gasoline):",
    gas['Fuel Consumption City (L/100km)'].mean())
print("Medijan (regular gasoline):",
    gas['Fuel Consumption City (L/100km)'].median())

#g
diesel4 = data[(data['Fuel Type'] == 'Z') & (data['Cylinders'] == 4)]

max_car = diesel4.loc[diesel4['Fuel Consumption City (L/100km)'].idxmax()]
print("\nDiesel vozilo s 4 cilindra i najvećom gradskom potrošnjom:")
print(max_car[['Make','Model','Fuel Consumption City (L/100km)']])


#h
manual = data[data['Transmission'].str.contains('M')]
print("\nBroj vozila s ručnim mjenjačem:", manual.shape[0])


#i
corr = data.corr(numeric_only=True)
print("\nKorelacijska matrica:")
print(corr)