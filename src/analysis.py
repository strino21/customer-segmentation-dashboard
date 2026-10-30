# analysis.py
# Prepara i dati dei clienti e li segmenta con K-Means

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Percorsi
DATA_DIR = Path(__file__).parent.parent / "data"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PROFILES_PATH = DATA_DIR / "profiles.csv"
IMAGES_DIR = Path(__file__).parent.parent / "immagini"

# Colonne numeriche del CSV, descritte nell'esplorazione iniziale
RAW_NUMERIC_COLUMNS = ["days_since_last", "recharges", "total_spent"]

# Variabili usate da K-Means: R (giorni dall'ultima ricarica), F (ricariche),
# M (spesa media per ricarica, calcolata in prepare_data)
CLUSTERING_FEATURES = ["days_since_last", "recharges", "avg_spent"]

# Etichette per i valori mancanti
NOT_SPECIFIED_ZONE = "Non indicata"
NO_PREDOMINANT_CATEGORY = "Nessuna prevalente"

# Parametri di K-Means, uguali in tutto il progetto (riproducibilità)
RANDOM_STATE = 42
N_INIT = 10
ELBOW_K_VALUES = range(1, 11)
SILHOUETTE_K_VALUES = range(2, 11)


def load_data() -> pd.DataFrame:
    """Legge customers.csv e restituisce un DataFrame, con signup_date come data."""
    df = pd.read_csv(CUSTOMERS_PATH, parse_dates=["signup_date"])
    return df


def explore(df: pd.DataFrame) -> None:
    """Stampa una panoramica dei dati: tipi, statistiche di R, F, M,
    valori mancanti e numero di clienti per zona e per categoria."""
    print(df.dtypes)
    print(df[RAW_NUMERIC_COLUMNS].describe())
    print(df.isna().sum())
    print(df["zone"].value_counts(dropna=False))
    print(df["category"].value_counts(dropna=False))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara i dati per il clustering.

    Sostituisce i valori mancanti di zona e categoria con un'etichetta esplicita,
    per non perdere clienti con R, F, M completi, e calcola la spesa media per
    ricarica (avg_spent), usata come variabile M al posto della spesa totale.
    Restituisce il DataFrame preparato.
    """
    df = df.fillna(value={"zone": NOT_SPECIFIED_ZONE, "category": NO_PREDOMINANT_CATEGORY})
    df["avg_spent"] = df["total_spent"] / df["recharges"]
    return df


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Normalizza R, F, M con lo z-score (media 0, deviazione standard 1).

    Restituisce i dati normalizzati e lo scaler addestrato, che serve
    per normalizzare allo stesso modo i nuovi clienti nella dashboard.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[CLUSTERING_FEATURES])
    return X, scaler


def compute_inertias(X: np.ndarray, k_values: range) -> list[float]:
    """Esegue K-Means per ogni k e restituisce l'inertia di ciascuno (metodo del gomito)."""
    inertias = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    return inertias


def compute_silhouettes(X: np.ndarray, k_values: range) -> list[float]:
    """Esegue K-Means per ogni k e restituisce la silhouette media di ciascuno."""
    silhouettes = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        silhouettes.append(silhouette)
    return silhouettes


def plot_elbow(k_values: range, inertias: list[float]) -> None:
    """Disegna il grafico del gomito (inertia per ogni k) e lo salva in immagini/gomito.png."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_values), inertias, marker="o")
    ax.set_xticks(list(k_values))
    ax.set_xlabel("Numero di gruppi (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Metodo del gomito")
    fig.savefig(IMAGES_DIR / "gomito.png", dpi=150, bbox_inches="tight")
    plt.close(fig) 


def plot_silhouette(k_values: range, silhouettes: list[float]) -> None:
    """Disegna la silhouette media per ogni k e la salva in immagini/silhouette.png."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_values), silhouettes, marker="o")
    ax.set_xticks(list(k_values))
    ax.set_xlabel("Numero di gruppi (k)")
    ax.set_ylabel("Silhouette media")
    ax.set_title("Silhouette media per numero di gruppi")
    fig.savefig(IMAGES_DIR / "silhouette.png", dpi=150, bbox_inches="tight")
    plt.close(fig) 


def main() -> None:
    """Prepara i dati, normalizza R, F, M e calcola gomito e silhouette per scegliere k."""
    np.set_printoptions(suppress=True)
    df = load_data()
    explore(df)
    df = prepare_data(df)

    X, scaler = scale_features(df)
    print("Medie imparate dallo scaler:",  np.round(scaler.mean_, 2))
    print("Medie dopo la normalizzazione:", np.round(X.mean(axis=0), 2))
    print("Deviazioni standard dopo la normalizzazione:", np.round(X.std(axis=0), 2))

    inertias = compute_inertias(X, ELBOW_K_VALUES)
    print("Inertia:", np.round(inertias,2))

    silhouettes = compute_silhouettes(X, SILHOUETTE_K_VALUES)
    print("Silhouette:", np.round(silhouettes,3))

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_elbow(ELBOW_K_VALUES, inertias)
    plot_silhouette(SILHOUETTE_K_VALUES, silhouettes)
    

if __name__ == "__main__":
    main()