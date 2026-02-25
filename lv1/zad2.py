try:
    ocjena = float(input("Unesite ocjenu (između 0.0 i 1.0): "))

    if ocjena < 0.0 or ocjena > 1.0:
        print("Broj mora biti u intervalu od 0.0 do 1.0.")
    else:
        if ocjena >= 0.9:
            print("A")
        elif ocjena >= 0.8:
            print("B")
        elif ocjena >= 0.7:
            print("C")
        elif ocjena >= 0.6:
            print("D")
        else:
            print("F")
except ValueError:
    print("Greška: niste unijeli broj.")