broj_rijeci = {"ham": [], "spam": []}
spam_usklicnik = 0

try:
    with open("C:/Users/student/Desktop/lv1/lv1/SMSSpamCollection.txt", "r", encoding="utf-8") as f:
        for red in f:
            red = red.strip()  
            if not red:
                continue 
            
            tip, *poruka = red.split()
            poruka_tekst = " ".join(poruka)

            broj = len(poruka_tekst.split())
            if tip in ["ham", "spam"]:
                broj_rijeci[tip].append(broj)

            if tip == "spam" and poruka_tekst.endswith("!"):
                spam_usklicnik += 1

    prosjek_ham = sum(broj_rijeci["ham"]) / len(broj_rijeci["ham"])
    prosjek_spam = sum(broj_rijeci["spam"]) / len(broj_rijeci["spam"])

    print(f"Prosječan broj riječi u ham porukama: {prosjek_ham:.2f}")
    print(f"Prosječan broj riječi u spam porukama: {prosjek_spam:.2f}")
    print(f"Broj spam poruka koje završavaju usklicnikom: {spam_usklicnik}")

except FileNotFoundError:
    print("Greška: datoteka 'SMSSpamCollection.txt' ne postoji.")