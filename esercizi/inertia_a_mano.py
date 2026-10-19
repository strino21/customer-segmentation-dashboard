import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2], [3, 2], [6, 5], [8, 7]])

km = KMeans(n_clusters=2, n_init=10, random_state=42)
km.fit(X)

# Calcolo a mano
totale = 0
for i in range(len(X)):
    cliente = X[i]
    centroide = km.cluster_centers_[km.labels_[i]]   # il centro del gruppo di questo cliente
    distanza2 = (cliente[0] - centroide[0])**2 + (cliente[1] - centroide[1])**2
    totale += distanza2

print("Inertia calcolata a mano:", totale)
print("Inertia di scikit-learn: ", km.inertia_)