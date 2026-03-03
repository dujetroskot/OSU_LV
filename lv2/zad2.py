import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("C:/Users/student/Desktop/lv2/OSU_LV/lv2/data.csv", delimiter=",", skiprows=1)

broj_osoba = data.shape[0]
print("a) Broj osoba:", broj_osoba)

plt.figure()
plt.scatter(data[:,1], data[:,2])
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase")
plt.show()

plt.figure()
plt.scatter(data[::50, 1], data[::50, 2])
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase (svaka 50-ta osoba)")
plt.show()

visine = data[:,1]
print("\nd) Statistika visine (svi):")
print("Minimalna visina:", np.min(visine))
print("Maksimalna visina:", np.max(visine))
print("Srednja visina:", np.mean(visine))


ind_m = (data[:,0] == 1)
ind_z = (data[:,0] == 0)
visine_m = data[ind_m, 1]
visine_z = data[ind_z, 1]

print("\ne) Statistika visine (muškarci):")
print("Min:", np.min(visine_m))
print("Max:", np.max(visine_m))
print("Mean:", np.mean(visine_m))

print("\nStatistika visine (žene):")
print("Min:", np.min(visine_z))
print("Max:", np.max(visine_z))
print("Mean:", np.mean(visine_z))