rijeci = {}

try:
    with open("C:/Users/student/Desktop/lv1/lv1/song.txt", "r", encoding="utf-8") as datoteka:
        for red in datoteka:
            red = red.lower()
            for znak in ".,!?;:\"()[]{}":
                red = red.replace(znak, "")

            lista_rijeci = red.split()

            for rijec in lista_rijeci:
                if rijec in rijeci:
                    rijeci[rijec] += 1
                else:
                    rijeci[rijec] = 1

    jednom = []

    for r, broj in rijeci.items():
        if broj == 1:
            jednom.append(r)

    print("Broj riječi koje se pojavljuju samo jednom:", len(jednom))
    print("Te riječi su:")
    for r in jednom:
        print(r)

except FileNotFoundError:
    print("Greška: datoteka 'song.txt' ne postoji.")