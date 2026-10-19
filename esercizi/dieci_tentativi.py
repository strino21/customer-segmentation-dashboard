import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.flatten()                    # 10 grafici in una lista

inertie = []

# Rifacciamo a mano i 10 tentativi di n_init=10, tutti con k = 3
for tentativo, ax in zip(range(10), axs):
    km = KMeans(n_clusters=3, init="random", n_init=1, random_state=10 + tentativo)
    gruppi = km.fit_predict(X)
    inertie.append(km.inertia_)
    ax.scatter(df["richieste"], df["spesa"], c=gruppi)
    ax.set_title(f"Tentativo {tentativo + 1}: inertia {km.inertia_:.1f}")

# Evidenziamo quello che n_init=10 terrebbe: l'inertia più bassa
migliore = inertie.index(min(inertie))
axs[migliore].set_title(f"Tentativo {migliore + 1}: inertia {inertie[migliore]:.1f} ← MIGLIORE",
                        color="green")

plt.show()