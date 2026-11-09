# app.py
# Dashboard Streamlit dei segmenti di clienti di AlveareAPI
# Avvio (dalla cartella del progetto): streamlit run src/app.py

import joblib
import pandas as pd
import streamlit as st

# Percorsi e variabili presi da analysis.py, così le due parti del progetto
# usano sempre gli stessi file e le stesse colonne
from analysis import (
    CLUSTERING_FEATURES,
    CLUSTERS_PATH,
    KMEANS_PATH,
    SCALER_PATH,
)

# Testi della pagina
PAGE_TITLE = "AlveareAPI - Segmentazione della clientela"
PAGE_DESCRIPTION = (
    "Gruppi di clienti individuati con K-Means su recency, "
    "frequenza delle ricariche e spesa media."
)

# Colonne del profilo e loro nomi in italiano, mostrati nella tabella
PROFILE_COLUMNS = {
    "days_since_last": "Giorni dall'ultima ricarica",
    "recharges": "Ricariche",
    "avg_spent": "Spesa media per ricarica (€)",
    "total_spent": "Spesa totale (€)",
}

# Regole per il nuovo cliente (le stesse del generatore dei dati)
MIN_RECHARGE_EUR = 10      # la ricarica minima è 10 €
MAX_DAYS = 365             # dati riferiti a 12 mesi di attività


@st.cache_data
def load_clusters() -> pd.DataFrame:
    """Legge data/clusters.csv: ogni cliente con il suo gruppo e il nome del segmento."""
    df = pd.read_csv(CLUSTERS_PATH, parse_dates=["signup_date"])
    return df


@st.cache_resource
def load_model():
    """Carica modello K-Means e scaler salvati da analysis.py (una volta sola)."""
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    return kmeans, scaler


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Mostra i filtri nella barra laterale e restituisce solo i clienti che li rispettano.

    Filtri richiesti dalla traccia: zona, periodo (data di iscrizione) e categoria di
    servizio preferita. I filtri non rifanno K-Means: mostrano un sottoinsieme dei
    clienti con i gruppi già assegnati.
    """
    st.sidebar.header("Filtri")

    zones = sorted(df["zone"].unique())
    selected_zones = st.sidebar.multiselect("Zona", options=zones, default=zones)

    categories = sorted(df["category"].unique())
    selected_categories = st.sidebar.multiselect(
        "Categoria di servizio preferita", options=categories, default=categories
    )

    first_day = df["signup_date"].min().date()
    last_day = df["signup_date"].max().date()
    period = st.sidebar.date_input(
        "Periodo di iscrizione",
        value=(first_day, last_day),
        min_value=first_day,
        max_value=last_day,
    )

    mask = df["zone"].isin(selected_zones) & df["category"].isin(selected_categories)
    if len(period) == 2:            # mentre si sceglie la seconda data ne arriva una sola
        start, end = period
        mask &= df["signup_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return df[mask]


def segment_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Restituisce il profilo di ogni segmento: medie delle variabili, numero e quota di clienti.

    I segmenti sono ordinati per giorni dall'ultima ricarica, dal più attivo al meno attivo.
    """
    groups = df.groupby("segment")
    profiles = groups[list(PROFILE_COLUMNS)].mean()
    profiles["customers"] = groups.size()
    profiles["share"] = profiles["customers"] / len(df) * 100
    return profiles.sort_values("days_since_last")


def show_scatter(df: pd.DataFrame) -> None:
    """Grafico a dispersione dei clienti: giorni dall'ultima ricarica e spesa media, per segmento."""
    st.subheader("I gruppi di clienti")
    # Colonna del segmento rinominata solo per il grafico, così la legenda è in italiano
    chart_df = df.rename(columns={"segment": "Segmento"})
    st.scatter_chart(
        chart_df,
        x="days_since_last",
        y="avg_spent",
        color="Segmento",
        x_label="Giorni dall'ultima ricarica",
        y_label="Spesa media per ricarica (€)",
    )


def show_profiles(profiles: pd.DataFrame) -> None:
    """Mostra i profili: un riquadro con la numerosità per segmento e la tabella delle medie."""
    st.subheader("Profilo dei gruppi")

    columns = st.columns(len(profiles))
    for column, (segment, row) in zip(columns, profiles.iterrows()):
        column.metric(segment, f"{row['customers']:.0f} clienti")
        column.caption(f"{row['share']:.0f}% dei clienti selezionati")

    table = profiles[list(PROFILE_COLUMNS)].rename(columns=PROFILE_COLUMNS).round(1)
    table.index.name = "Segmento"
    st.dataframe(table)
    st.caption("Valori medi di ogni segmento, calcolati sui dati originali (non normalizzati).")


def classify_new_customer(df: pd.DataFrame) -> None:
    """Modulo per inserire un nuovo cliente e ottenere il suo segmento.

    Il cliente viene normalizzato con lo scaler addestrato sui 400 clienti (transform,
    mai fit) e assegnato al gruppo con il modello K-Means salvato (predict).
    """
    st.subheader("Classifica un nuovo cliente")

    with st.form("new_customer"):
        days_since_last = st.number_input(
            "Giorni dall'ultima ricarica", min_value=0, max_value=MAX_DAYS, value=30, step=1
        )
        recharges = st.number_input(
            "Numero di ricariche", min_value=1, max_value=100, value=3, step=1
        )
        total_spent = st.number_input(
            "Spesa totale (€)", min_value=float(MIN_RECHARGE_EUR), value=100.0, step=10.0
        )
        submitted = st.form_submit_button("Assegna al gruppo")

    if not submitted:
        return

    # Regola di coerenza: almeno 10 € per ogni ricarica
    if total_spent < MIN_RECHARGE_EUR * recharges:
        st.error(
            f"Con {recharges} ricariche la spesa totale deve essere almeno "
            f"{MIN_RECHARGE_EUR * recharges} € (ricarica minima {MIN_RECHARGE_EUR} €)."
        )
        return

    kmeans, scaler = load_model()
    new_customer = pd.DataFrame([{
        "days_since_last": days_since_last,
        "recharges": recharges,
        "avg_spent": total_spent / recharges,
    }])[CLUSTERING_FEATURES]
    cluster = int(kmeans.predict(scaler.transform(new_customer))[0])

    # Dal numero del gruppo al nome del segmento, come in clusters.csv
    segment_names = df.drop_duplicates("cluster").set_index("cluster")["segment"]
    st.success(f"Il nuovo cliente appartiene al segmento: **{segment_names[cluster]}**")
    st.caption(f"Spesa media per ricarica calcolata: {total_spent / recharges:.2f} €")


def main() -> None:
    """Costruisce la dashboard: filtri, grafico dei gruppi, profili e classificazione."""
    st.set_page_config(page_title="AlveareAPI", layout="wide")
    st.title(PAGE_TITLE)
    st.write(PAGE_DESCRIPTION)

    if not (CLUSTERS_PATH.exists() and KMEANS_PATH.exists() and SCALER_PATH.exists()):
        st.error("Risultati non trovati: esegui prima `python src/analysis.py`.")
        st.stop()

    df = load_clusters()
    filtered = sidebar_filters(df)
    st.write(f"Clienti selezionati: **{len(filtered)}** su {len(df)}")

    if filtered.empty:
        st.warning("Nessun cliente corrisponde ai filtri scelti.")
    else:
        show_scatter(filtered)
        show_profiles(segment_profiles(filtered))

    classify_new_customer(df)


if __name__ == "__main__":
    main()
