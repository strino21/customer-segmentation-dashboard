import random
import pandas as pd

def spesa_media(clienti):
    totale = 0
    for cliente in clienti:
        totale += cliente["spesa"]
    return totale / len(clienti)

def spesa_media_ez(clienti):
    return sum(c(["spesa"]) for c in clienti) / len(clienti)

clienti = []

for i in range(10):
    zona = random.choice(["Nord", "Centro", "Sud"])
    richieste = random.randint(1, 30)
    spesa = round(random.uniform(10, 500),2)

    cliente = {"id": i, "zona": zona, "richieste": richieste, "spesa": spesa}
    clienti.append(cliente)
    print(cliente)

media = spesa_media(clienti)
print(f"La spesa media dei clienti è: {media:.2f}€")


df = pd.DataFrame(clienti)      # la lista di dizionari diventa una tabella
print(df["spesa"].mean())
print(df)
print("=======================================")
print(df.describe())   
df.to_csv("clienti.csv", index=False)

df2 = pd.read_csv("clienti.csv")
print(df2.head())         # le prime 5 righe