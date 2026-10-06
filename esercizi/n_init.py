import random
import pandas as pd
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

for tentativi in [1, 10]:
    km = KMeans(n_clusters=3, init="random", n_init=tentativi, random_state=12)
    df["cluster"] = km.fit_predict(X)
    print(f"--- n_init = {tentativi} ---")
    print("Clienti per gruppo:", df.groupby("cluster").size().tolist())
    print("Inertia:", round(km.inertia_, 1))
    print()