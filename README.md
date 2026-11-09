# AlveareAPI – Segmentazione della clientela con K-Means

Prototipo del Project Work **PW 28** (Università Pegaso, corso di laurea L-31):
segmentazione automatica dei clienti di una startup digitale di servizi con
apprendimento automatico non supervisionato (K-Means) e dashboard interattiva.

**AlveareAPI** è una startup immaginaria: una piattaforma italiana di API a consumo
(email, OCR, geolocalizzazione, traduzione) con crediti prepagati. Tutti i dati sono
**sintetici**, generati per l'occasione, e non contengono informazioni personali.

## Cosa fa

1. **Genera** 400 clienti realistici in 4 profili (molto attivi, occasionali, recenti, inattivi).
2. **Analizza** i dati: gestione dei valori mancanti, normalizzazione, scelta del numero di
   gruppi con metodo del gomito e silhouette, K-Means con k = 3.
3. **Mostra** i risultati in una dashboard Streamlit: grafico dei gruppi, profilo di ogni
   gruppo, filtri per zona, periodo e categoria, classificazione di un nuovo cliente.

## Requisiti

- Python 3.11 o successivo
- Le librerie in `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, streamlit, joblib)

## Installazione

Dalla cartella del progetto:

```bash
python -m venv .venv
```

Attivazione dell'ambiente virtuale:

- macOS / Linux: `source .venv/bin/activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`

Installazione delle librerie:

```bash
pip install -r requirements.txt
```

## Esecuzione

I risultati sono già inclusi nel repository, quindi la dashboard si può avviare subito:

```bash
streamlit run src/app.py
```

Per rifare tutto da zero, nell'ordine:

```bash
python src/generate_data.py   # crea data/customers.csv e data/profiles.csv
python src/analysis.py        # crea data/clusters.csv, models/, immagini/
streamlit run src/app.py      # apre la dashboard nel browser (http://localhost:8501)
```

Con gli stessi parametri i file generati sono **identici** a ogni esecuzione.

## Parametri di riproducibilità

| Parametro | Valore | Dove |
|---|---|---|
| Seed del generatore | 42 | `src/generate_data.py` (`SEED`) |
| Data di riferimento dei dati | 31/10/2026 | `src/generate_data.py` (`EXTRACTION_DATE`) |
| Numero di clienti | 400 | `src/generate_data.py` (`TOTAL_CUSTOMERS`) |
| Variabili del clustering | giorni dall'ultima ricarica, numero di ricariche, spesa media per ricarica | `src/analysis.py` (`CLUSTERING_FEATURES`) |
| Normalizzazione | z-score (`StandardScaler`) | `src/analysis.py` |
| K-Means | `n_init = 10`, `random_state = 42` | `src/analysis.py` |
| Numero di gruppi | k = 3 (confrontato con k = 4) | `src/analysis.py` (`CHOSEN_K`) |

## Risultati

| Segmento | Clienti | Giorni dall'ultima ricarica | Ricariche | Spesa media per ricarica |
|---|---|---|---|---|
| Clienti di valore | 60 | 10,8 | 25,4 | 192,0 € |
| Occasionali e nuovi | 229 | 59,1 | 3,5 | 40,5 € |
| Inattivi | 111 | 178,9 | 5,6 | 167,9 € |

Silhouette media con k = 3: 0,500. I grafici del gomito e della silhouette sono in `immagini/`.

Verifica con i profili usati per generare i dati (`profiles.csv`, non usato dal clustering):
K-Means ritrova tutti i clienti molto attivi; i clienti recenti finiscono con gli occasionali,
perché nei comportamenti di acquisto (R, F, M) sono simili.

## Struttura del progetto

```
├── data/
│   ├── customers.csv      # dati sintetici dei clienti (una riga per cliente)
│   ├── profiles.csv       # profilo con cui è stato generato ogni cliente (verifica finale in analysis.py)
│   └── clusters.csv       # risultati: ogni cliente con il suo gruppo e segmento
├── immagini/              # grafici del gomito e della silhouette
├── models/                # modello K-Means e scaler salvati (usati dalla dashboard)
├── src/
│   ├── generate_data.py   # generatore dei dati sintetici
│   ├── analysis.py        # preparazione, scelta di k, clustering, salvataggio
│   └── app.py             # dashboard Streamlit
├── requirements.txt
├── LICENSE
└── README.md
```

## Colonne di `customers.csv`

| Colonna | Significato |
|---|---|
| `customer_id` | identificativo del cliente |
| `zone` | ripartizione geografica ISTAT (Nord-Ovest, Nord-Est, Centro, Sud, Isole) |
| `category` | categoria di servizio preferita (Email, OCR, Geolocalizzazione, Traduzione) |
| `signup_date` | data di iscrizione |
| `days_since_last` | giorni dall'ultima ricarica di crediti |
| `recharges` | numero di ricariche nei 12 mesi |
| `total_spent` | spesa totale in euro |

Zona e categoria contengono alcuni valori mancanti voluti (2% e 3%), gestiti nell'analisi.

## Licenza

Distribuito con licenza MIT: vedi il file `LICENSE`.
