import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Dati: 90 clienti con 3 profili ---
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

# --- Tre centroidi di partenza scelti male apposta ---
partenza = np.array([[0.0, -1.5], [0.2, -1.4], [0.4, -1.6]])

fig, assi = plt.subplots(1, 4, figsize=(16, 4))

# Pannello 0: partenza
assi[0].scatter(X[:, 0], X[:, 1], c="lightgray")
assi[0].scatter(partenza[:, 0], partenza[:, 1], marker="X", s=200, c="red")
assi[0].set_title("Partenza")

# Pannelli 1-3: dopo 1, 2 e 3 iterazioni
for n, ax in zip([1, 2, 3], assi[1:]):
    km = KMeans(n_clusters=3, init=partenza, n_init=1, max_iter=n)
    gruppi = km.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=gruppi)
    c = km.cluster_centers_
    ax.scatter(c[:, 0], c[:, 1], marker="X", s=200, c="red")
    ax.set_title(f"Dopo {n} iterazioni")

plt.show()