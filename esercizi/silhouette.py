import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

random.seed(42)
clienti = []
for i in range(30):
    clienti.append({"richieste": random.randint(20, 30), "spesa": random.uniform(350, 500)})
for i in range(30):
    clienti.append({"richieste": random.randint(1, 6), "spesa": random.uniform(10, 120)})
for i in range(30):
    clienti.append({"richieste": random.randint(2, 8), "spesa": random.uniform(300, 450)})
df = pd.DataFrame(clienti)
X = StandardScaler().fit_transform(df[["richieste", "spesa"]])

valori_k = range(2, 9)          # da 2: con un solo gruppo non c'è un "gruppo accanto"
silhouette = []

for k in valori_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    gruppi = km.fit_predict(X)                 # il gruppo di ogni cliente
    s = silhouette_score(X, gruppi)            # media dei voti di tutti i clienti
    silhouette.append(s)
    print(f"k = {k}: silhouette = {s:.3f}")

plt.plot(valori_k, silhouette, marker="o")
plt.xlabel("Numero di gruppi (k)")
plt.ylabel("Silhouette media")
plt.title("Silhouette al variare di k")
plt.show()