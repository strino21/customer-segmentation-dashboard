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

valori_k = range(1, 9)      # k da 1 a 8
inertie = []                # qui salverai l'inertia di ogni k

for k in valori_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertie.append(km.inertia_)
    print(f"k = {k}: inertia = {km.inertia_:.1f}")

plt.plot(valori_k, inertie, marker="o")
plt.xlabel("Numero di gruppi (k)")
plt.ylabel("Inertia")
plt.title("Metodo del gomito")
plt.show()