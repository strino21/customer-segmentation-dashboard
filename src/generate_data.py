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

# Intervalli (minimo, massimo) di ogni profilo – valori definiti in startup.md
PROFILE_RANGES = {
    "very_active": {
        "days_since_signup": (120, 365),   # servono mesi per arrivare a 15-40 ricariche
        "days_since_last":   (0, 20),      # R: ricaricano di continuo
        "recharges":         (15, 40),     # F
        "total_spent":       (1500, 8000), # M
    },
    "occasional": {
        "days_since_signup": (30, 365),    # iscritti da almeno un mese
        "days_since_last":   (30, 120),    # tornano ogni tanto
        "recharges":         (2, 6),
        "total_spent":       (30, 250),
    },
    "recent": {
        "days_since_signup": (0, 59),      # iscritti negli ultimi 60 giorni
        "days_since_last":   (0, 30),      # hanno appena iniziato
        "recharges":         (1, 3),
        "total_spent":       (10, 100),
    },
    "inactive": {
        "days_since_signup": (120, 365),   # almeno quanto la loro inattività
        "days_since_last":   (120, 365),   # fermi da almeno 4 mesi
        "recharges":         (1, 10),      # in passato potevano essere anche buoni clienti
        "total_spent":       (20, 1000),
    },
}



