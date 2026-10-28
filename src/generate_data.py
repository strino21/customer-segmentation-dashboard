# generate_data.py
# Genera un dataset sintetico di clienti della startup e lo salva in data/customers.csv

import random
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

SEED = 42
EXTRACTION_DATE = date(2026, 10, 31)  
ZONES = ["Nord-Ovest", "Nord-Est", "Centro", "Sud", "Isole"]   # ripartizioni ISTAT
CATEGORIES = ["Email", "OCR", "Geolocalizzazione", "Traduzione"]

TOTAL_CUSTOMERS = 400

PROFILES = {
    "very_active": 60,
    "occasional": 160,
    "recent": 60,
    "inactive": 120,
}
assert sum(PROFILES.values()) == TOTAL_CUSTOMERS, (
    f"I profili sommano {sum(PROFILES.values())}, attesi {TOTAL_CUSTOMERS}"
)

DATA_DIR = Path(__file__).parent.parent / "data"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PROFILES_PATH = DATA_DIR / "profiles.csv"

# Percentuali di valori mancanti voluti
MISSING_ZONE_RATE = 0.02
MISSING_CATEGORY_RATE = 0.03

# Intervalli (minimo, massimo) di ogni profilo
PROFILE_RANGES = {
    "very_active": {
        "days_since_signup": (120, 365),   # servono mesi per arrivare a 15-40 ricariche
        "days_since_last":   (0, 20),      # R: ricaricano di continuo
        "recharges":         (15, 40),     # F
        "total_spent":       (1500, 8000), # M
    },
    "occasional": {
        "days_since_signup": (30, 365),
        "days_since_last":   (30, 120),
        "recharges":         (2, 6),
        "total_spent":       (30, 250),
    },
    "recent": {
        "days_since_signup": (0, 59),      # iscritti negli ultimi 60 giorni
        "days_since_last":   (0, 30),
        "recharges":         (1, 3),
        "total_spent":       (10, 100),
    },
    "inactive": {
        "days_since_signup": (120, 365),
        "days_since_last":   (120, 365),   # fermi da almeno 4 mesi
        "recharges":         (1, 10),
        "total_spent":       (20, 1000),
    },
}


def generate_customer(customer_id: int, profile: str) -> dict:
    """Genera un cliente casuale del profilo indicato.

    I valori vengono estratti dagli intervalli in PROFILE_RANGES, rispettando
    le regole di coerenza: l'ultima ricarica non può essere precedente
    all'iscrizione e la spesa totale è almeno 10 € per ogni ricarica.
    Restituisce un dizionario con i dati del cliente.
    """
    ranges = PROFILE_RANGES[profile]

    min_days_since_signup, max_days_since_signup = ranges["days_since_signup"]
    days_since_signup = random.randint(min_days_since_signup, max_days_since_signup)

    # Regola 1: non si può ricaricare prima di essersi iscritti
    min_days_since_last, max_days_since_last = ranges["days_since_last"]
    max_days_since_last = min(max_days_since_last, days_since_signup)
    days_since_last = random.randint(min_days_since_last, max_days_since_last)

    min_recharges, max_recharges = ranges["recharges"]
    recharges = random.randint(min_recharges, max_recharges)

    # Regola 2: ricarica minima di 10 €, quindi spesa >= 10 € x ricariche
    min_total_spent, max_total_spent = ranges["total_spent"]
    min_total_spent = max(min_total_spent, 10 * recharges)
    total_spent = round(random.uniform(min_total_spent, max_total_spent), 2)

    signup_date = EXTRACTION_DATE - timedelta(days=days_since_signup)

    customer = {
        "customer_id": customer_id,
        "profile": profile,
        "zone": random.choice(ZONES),
        "category": random.choice(CATEGORIES),
        "signup_date": signup_date,
        "days_since_last": days_since_last,
        "recharges": recharges,
        "total_spent": total_spent,
    }
    return customer


def generate_dataset() -> pd.DataFrame:
    """Genera tutti i clienti del dataset, profilo per profilo.

    Il numero di clienti per profilo è preso da PROFILES; gli id sono
    assegnati in sequenza da 1 al totale.
    Restituisce un DataFrame con una riga per cliente.
    """
    customers = []
    customer_id = 1
    for profile, count in PROFILES.items():
        for _ in range(count):
            customers.append(generate_customer(customer_id, profile))
            customer_id += 1
    df = pd.DataFrame(customers)
    return df


def add_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Rende vuote alcune zone e categorie, per simulare dati reali incompleti.

    Svuota il 2% delle zone e il 3% delle categorie, scegliendo le righe a caso
    in modo riproducibile. Le due estrazioni usano seed diversi per essere
    indipendenti. Restituisce il DataFrame modificato.
    """
    n_missing_zone = round(MISSING_ZONE_RATE * len(df))
    n_missing_category = round(MISSING_CATEGORY_RATE * len(df))

    zone_rows = df.sample(n=n_missing_zone, random_state=SEED).index
    category_rows = df.sample(n=n_missing_category, random_state=SEED + 1).index

    df.loc[zone_rows, "zone"] = None
    df.loc[category_rows, "category"] = None
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Controlla che il dataset generato rispetti tutte le regole prima del salvataggio.

    Verifica numero di righe, id unici, colonne, clienti per profilo,
    regole di coerenza e valori mancanti. Se una regola non è rispettata,
    il programma si ferma con un AssertionError.
    """
    expected_columns = ["customer_id", "profile", "zone", "category",
                        "signup_date", "days_since_last", "recharges", "total_spent"]

    assert len(df) == TOTAL_CUSTOMERS, f"Attese {TOTAL_CUSTOMERS} righe, trovate {len(df)}"

    assert df["customer_id"].is_unique, "Ci sono customer_id duplicati"

    assert list(df.columns) == expected_columns, f"Colonne inattese: {list(df.columns)}"

    assert df["profile"].value_counts().to_dict() == PROFILES, "Conteggio per profilo errato"

    assert (df["total_spent"] >= 10 * df["recharges"]).all(), "Spesa inferiore a 10 € per ricarica"

    assert df["zone"].isna().sum() == round(MISSING_ZONE_RATE * len(df)), \
        "Numero di zone mancanti errato"
    assert df["category"].isna().sum() == round(MISSING_CATEGORY_RATE * len(df)), \
        "Numero di categorie mancanti errato"

    for column in ["days_since_last", "recharges", "total_spent"]:
        assert df[column].isna().sum() == 0, f"Valori mancanti in {column}"

    days_since_signup = (pd.Timestamp(EXTRACTION_DATE) - pd.to_datetime(df["signup_date"])).dt.days
    assert (df["days_since_last"] <= days_since_signup).all(), \
        "Ultima ricarica precedente all'iscrizione"

    print("Validazione superata")


def main() -> None:
    """Genera il dataset, lo valida e lo salva nella cartella data/.

    customers.csv contiene i dati dei clienti senza il profilo; profiles.csv
    contiene solo id e profilo, da usare per verificare i cluster trovati.
    """
    random.seed(SEED)
    df = generate_dataset()
    df = add_missing(df)
    validate_dataset(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df[["customer_id", "profile"]].to_csv(PROFILES_PATH, index=False)
    df = df.drop(columns=["profile"])
    df.to_csv(CUSTOMERS_PATH, index=False)
    print(f"Salvati {len(df)} clienti in {CUSTOMERS_PATH}")


if __name__ == "__main__":
    main()