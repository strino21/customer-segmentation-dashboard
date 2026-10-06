import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

random.seed(42)   # rende i numeri casuali sempre uguali → risultato riproducibile

clienti = []

# Profilo 1: fedeli, tante richieste e spesa alta
for i in range(30):
    clienti.append({"richieste": random.randint(20, 30), "spesa": random.uniform(350, 500)})

# Profilo 2: occasionali, poche richieste e spesa bassa
for i in range(30):
    clienti.append({"richieste": random.randint(1, 6), "spesa": random.uniform(10, 120)})

# Profilo 3: "grandi acquisti", poche richieste ma spesa alta
for i in range(30):
    clienti.append({"richieste": random.randint(2, 8), "spesa": random.uniform(300, 450)})

df = pd.DataFrame(clienti)

# 1. Normalizzazione: porta richieste e spesa sulla stessa scala
X = StandardScaler().fit_transform(df[["richieste", "spesa"]])
#X = df[["richieste", "spesa"]]

# 2. K-Means: chiediamo 3 gruppi
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# 3. Profilo di ogni gruppo
print(df.groupby("cluster").mean().round(1))
print(df.groupby("cluster").size())

# 4. Grafico: ogni punto è un cliente, il colore è il gruppo trovato
plt.scatter(df["richieste"], df["spesa"], c=df["cluster"])
plt.xlabel("Numero di richieste")
plt.ylabel("Spesa (€)")
plt.title("Clienti raggruppati da K-Means")
plt.show()