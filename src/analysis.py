# analysis.py
# Prepara i dati dei clienti e li segmenta con K-Means

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Percorsi
DATA_DIR = Path(__file__).parent.parent / "data"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PROFILES_PATH = DATA_DIR / "profiles.csv"
CLUSTERS_PATH = DATA_DIR / "clusters.csv"
IMAGES_DIR = Path(__file__).parent.parent / "immagini"
MODELS_DIR = Path(__file__).parent.parent / "models"
KMEANS_PATH = MODELS_DIR / "kmeans.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

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
COMPARED_K_VALUES = [3, 4]

# Numero di gruppi scelto: la silhouette è quasi uguale per k = 3 (0,500) e k = 4 (0,499),
# ma con k = 4 si stacca un gruppo di soli 15 clienti, troppo piccolo per azioni di marketing
CHOSEN_K = 3

# Nomi dei segmenti, assegnati in name_segments guardando il profilo di ogni gruppo
SEGMENT_HIGH_VALUE = "Clienti di valore"
SEGMENT_OCCASIONAL = "Occasionali e nuovi"
SEGMENT_INACTIVE = "Inattivi"


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


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    """Addestra K-Means con k gruppi sui dati normalizzati e restituisce il modello."""
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    kmeans.fit(X)
    return kmeans


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Restituisce il profilo di ogni gruppo: medie delle variabili originali e numero di clienti."""
    profile_columns = CLUSTERING_FEATURES + ["total_spent"]
    df_with_clusters = df.assign(cluster=labels)
    groups = df_with_clusters.groupby("cluster")
    profiles = groups[profile_columns].mean()
    profiles["customers"] = groups.size()
    return profiles


def name_segments(profiles: pd.DataFrame) -> dict[int, str]:
    """Assegna un nome a ogni gruppo in base al suo profilo (pensato per CHOSEN_K = 3).

    I numeri dei gruppi li sceglie K-Means, quindi i nomi non vanno scritti a mano:
    il gruppo con l'ultima ricarica più recente è quello di valore, quello con
    l'ultima ricarica più lontana è quello degli inattivi, il restante gli occasionali.
    Restituisce un dizionario {numero del gruppo: nome}.
    """
    assert len(profiles) == 3, "name_segments assegna i nomi solo per 3 gruppi"
    by_recency = profiles["days_since_last"].sort_values()
    return {
        int(by_recency.index[0]): SEGMENT_HIGH_VALUE,
        int(by_recency.index[1]): SEGMENT_OCCASIONAL,
        int(by_recency.index[2]): SEGMENT_INACTIVE,
    }


def save_results(df: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler,
                 segment_names: dict[int, str]) -> None:
    """Salva i risultati del clustering e gli oggetti che servono alla dashboard.

    data/clusters.csv: ogni cliente con il numero e il nome del suo gruppo.
    models/kmeans.joblib e models/scaler.joblib: modello e scaler addestrati, per
    assegnare un nuovo cliente a un gruppo senza rifare l'addestramento.
    """
    results = df.assign(cluster=kmeans.labels_)
    results["avg_spent"] = results["avg_spent"].round(2)
    results["segment"] = results["cluster"].map(segment_names)
    results.to_csv(CLUSTERS_PATH, index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, KMEANS_PATH)
    joblib.dump(scaler, SCALER_PATH)


def main() -> None:
    """Esegue l'analisi completa: preparazione, scelta di k, modello finale e salvataggio."""
    np.set_printoptions(suppress=True)
    pd.set_option("display.width", 120)   # tabelle larghe stampate senza "..."
    df = load_data()
    explore(df)
    df = prepare_data(df)

    X, scaler = scale_features(df)
    print("Medie imparate dallo scaler:", np.round(scaler.mean_, 2))
    print("Medie dopo la normalizzazione:", np.round(X.mean(axis=0), 2))
    print("Deviazioni standard dopo la normalizzazione:", np.round(X.std(axis=0), 2))

    inertias = compute_inertias(X, ELBOW_K_VALUES)
    print("Inertia:", np.round(inertias, 2))

    silhouettes = compute_silhouettes(X, SILHOUETTE_K_VALUES)
    print("Silhouette:", np.round(silhouettes, 3))

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_elbow(ELBOW_K_VALUES, inertias)
    plot_silhouette(SILHOUETTE_K_VALUES, silhouettes)

    # Confronto tra le configurazioni candidate (k = 3 e k = 4)
    for k in COMPARED_K_VALUES:
        model = fit_kmeans(X, k)
        silhouette = silhouette_score(X, model.labels_)
        print(f"\n=== k = {k} · silhouette {silhouette:.3f} ===")
        print(cluster_profiles(df, model.labels_).round(1))

    # Modello finale con il k scelto
    final_model = fit_kmeans(X, CHOSEN_K)
    final_profiles = cluster_profiles(df, final_model.labels_)
    segment_names = name_segments(final_profiles)
    print(f"\n=== Modello finale: k = {CHOSEN_K} ===")
    print(final_profiles.rename(index=segment_names).round(1))

    save_results(df, final_model, scaler, segment_names)
    print(f"Risultati salvati in {CLUSTERS_PATH}, modello e scaler in {MODELS_DIR}")


if __name__ == "__main__":
    main()