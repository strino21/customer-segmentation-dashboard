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

configurazioni = [3, 4]                        # le due configurazioni da confrontare
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for k, ax in zip(configurazioni, axs):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    df["cluster"] = km.fit_predict(X)          # il gruppo di ogni cliente
    s = silhouette_score(X, df["cluster"])

    # Profilo dei gruppi: medie (valori originali) + quanti clienti
    profilo = df.groupby("cluster")[["richieste", "spesa"]].mean().round(1)
    profilo["clienti"] = df.groupby("cluster").size()

    print(f"\n===== k = {k}  |  silhouette = {s:.3f} =====")
    print(profilo)

    ax.scatter(df["richieste"], df["spesa"], c=df["cluster"])
    ax.set_title(f"k = {k}  (silhouette {s:.3f})")
    ax.set_xlabel("Numero di richieste")
    ax.set_ylabel("Spesa (€)")

plt.show()