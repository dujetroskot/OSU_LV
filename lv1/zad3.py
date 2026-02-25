brojevi = []

while True:
    unos = input("Unesite broj (ili 'Done' za kraj): ")

    if unos == "Done":
        break

    try:
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print("Greška: unos nije broj, pokušajte ponovno.")
        continue

if len(brojevi) == 0:
    print("Niste unijeli nijedan broj.")
else:
    print("\nBroj unesenih brojeva:", len(brojevi))
    print("Srednja vrijednost:", sum(brojevi) / len(brojevi))
    print("Minimalna vrijednost:", min(brojevi))
    print("Maksimalna vrijednost:", max(brojevi))

    brojevi.sort()
    print("Sortirana lista:", brojevi)