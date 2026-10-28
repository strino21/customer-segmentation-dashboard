from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PROFILES_PATH = DATA_DIR / "profiles.csv"


def load_data() -> pd.DataFrame:
    """Legge customers.csv e restituisce un DataFrame, con signup_date come data."""
    df = pd.read_csv(CUSTOMERS_PATH, parse_dates=["signup_date"])
    return df


def explore(df: pd.DataFrame) -> None:
    """Stampa una panoramica dei dati: tipi, statistiche di R, F, M,
    valori mancanti e numero di clienti per zona e per categoria."""
    print(df.dtypes)
    print(df[["days_since_last", "recharges", "total_spent"]].describe())
    print(df.isna().sum())
    print(df["zone"].value_counts(dropna=False))
    print(df["category"].value_counts(dropna=False))


def main() -> None:
    """Carica i dati dei clienti ed esegue l'esplorazione iniziale."""
    df = load_data()
    explore(df)


if __name__ == "__main__":
    main()