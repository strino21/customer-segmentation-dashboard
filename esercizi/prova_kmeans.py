import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

random.seed(42)   # numeri casuali sempre uguali → risultato riproducibile

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
#X = df[["richieste", "spesa"]])
# Esperimento senza normalizzazione: commenta la riga sopra e usa questa
# X = df[["richieste", "spesa"]]

# 2. K-Means
k = 3   # prova anche 2 e 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# 3. Profilo di ogni gruppo (valori originali, non normalizzati)
print(df.groupby("cluster").mean().round(1))
print(df.groupby("cluster").size())

# 4. Grafico
plt.scatter(df["richieste"], df["spesa"], c=df["cluster"])
plt.xlabel("Numero di richieste")
plt.ylabel("Spesa (€)")
plt.title(f"Clienti raggruppati da K-Means (k = {k})")
plt.show()