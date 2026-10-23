# generate_data.py
# Genera un dataset sintetico di clienti della startup e lo salva in data/customers.csv


import random
from datetime import date, timedelta
import pandas as pd

SEED = 42
EXTRACTION_DATE = date(2026, 10, 31)   # "oggi" del dataset: dopo 12 mesi di attività

ZONES = ["Nord-Ovest", "Nord-Est", "Centro", "Sud", "Isole"]   # ripartizioni ISTAT
CATEGORIES = ["Email", "OCR", "Geolocalizzazione", "Traduzione"]

PROFILES = {
    "very_active": 60,
    "occasional": 160,
    "recent": 60,
    "inactive": 120,
}

TOTAL_CUSTOMERS = sum(PROFILES.values())
assert TOTAL_CUSTOMERS == 400, f"Attesi 400 clienti, trovati {TOTAL_CUSTOMERS}"

# Intervalli (minimo, massimo) di ogni profilo
PROFILE_RANGES = {
    "very_active": {
        "days_since_signup": (120, 365),
        "days_since_last":   (0, 20),
        "recharges":         (15, 40),
        "total_spent":       (1500, 8000),
    },
    "occasional": {
        "days_since_signup": (30, 365),
        "days_since_last":   (30, 120),
        "recharges":         (2, 6),
        "total_spent":       (30, 250),
    },
    "recent": {
        "days_since_signup": (0, 59),
        "days_since_last":   (0, 30),
        "recharges":         (1, 3),
        "total_spent":       (10, 100),
    },
    "inactive": {
        "days_since_signup": (120, 365),
        "days_since_last":   (120, 365),
        "recharges":         (1, 10),
        "total_spent":       (20, 1000),
    },
}

def generate_customer(customer_id, profile):
    """Genera un cliente casuale indicizzato del profilo indicato.

    I valori vengono estratti dagli intervalli in PROFILE_RANGES, rispettando
    le regole di coerenza: l'ultima ricarica non può essere precedente
    all'iscrizione e la spesa totale è almeno 10 € per ogni ricarica.
    Restituisce un dizionario con i dati del cliente.
    """
    ranges = PROFILE_RANGES[profile]

    min_days_since_signup, max_days_since_signup = ranges["days_since_signup"]
    days_since_signup = random.randint(min_days_since_signup, max_days_since_signup)

    min_days_since_last, max_days_since_last = ranges["days_since_last"]
    max_days_since_last = min(max_days_since_last, days_since_signup)
    days_since_last = random.randint(min_days_since_last, max_days_since_last)

    min_recharges, max_recharges = ranges["recharges"]
    recharges = random.randint(min_recharges, max_recharges)

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



def generate_dataset():
    """Genera tutti i clienti del dataset per ogni profilo esistente.

    Il numero di clienti per profilo è preso da PROFILES; gli id sono
    assegnati in sequenza da 1 al totale.
    """
    customers = []
    customer_id = 1
    for profile, count in PROFILES.items():
        for _ in range(count):
            customers.append(generate_customer(customer_id, profile))
            customer_id += 1
    df = pd.DataFrame(customers)
    return df



# Prova rapida: un cliente per profilo
random.seed(SEED)
for customer_id, profile in enumerate(PROFILES, start=1):
    print(generate_customer(customer_id, profile))

random.seed(SEED)
df = generate_dataset()
print(df.head())
print(len(df))
print(df["profile"].value_counts())

print(df["customer_id"].min(), df["customer_id"].max())   # atteso: 1 400
print(df["customer_id"].is_unique)                         # atteso: True

print(df.to_string())	
