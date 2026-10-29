from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PROFILES_PATH = DATA_DIR / "profiles.csv"
CLUSTERING_FEATURES = ["days_since_last", "recharges", "total_spent"]
NOT_SPECIFIED_ZONE = "Non indicata"
NO_PREDOMINANT_CATEGORY = "Nessuna prevalente"

def load_data() -> pd.DataFrame:
    """Legge customers.csv e restituisce un DataFrame, con signup_date come data."""
    df = pd.read_csv(CUSTOMERS_PATH, parse_dates=["signup_date"])
    return df


def explore(df: pd.DataFrame) -> None:
    """Stampa una panoramica dei dati: tipi, statistiche di R, F, M,
    valori mancanti e numero di clienti per zona e per categoria."""
    print(df.dtypes)
    print(df[CLUSTERING_FEATURES].describe())
    print(df.isna().sum())
    print(df["zone"].value_counts(dropna=False))
    print(df["category"].value_counts(dropna=False))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Gestisce i valori mancanti di zona e categoria.

    I mancanti vengono sostituiti con un'etichetta esplicita invece di
    eliminare le righe, per non perdere clienti con R, F, M completi.
    Restituisce il DataFrame senza valori mancanti.
    """
    df = df.fillna(value={"zone": NOT_SPECIFIED_ZONE, "category": NO_PREDOMINANT_CATEGORY})
    return df


def main() -> None:
    """Carica i dati dei clienti, li esplora e li prepara per il clustering."""
    df = load_data()
    explore(df)
    df = prepare_data(df)
    explore(df)


if __name__ == "__main__":
    main()