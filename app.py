import streamlit as st
import pandas as pd
import penaltyblog as pb
import numpy as np
from scipy.stats import poisson
import soccerdata as sd

# Impostazioni della pagina web
st.set_page_config(page_title="Calcio Betting Model", page_icon="⚽", layout="wide")

# --- FUNZIONE PER CARICARE E PULIRE I DATI DEI GIOCATORI ---
@st.cache_data(show_spinner=False)
def carica_stats_giocatori():
    # Inizializza lo scraper per la Premier League 2023-24
    fbref = sd.FBref(leagues="ENG-Premier League", seasons="2023-2024")
    
    # Scarica le statistiche di tiro (tiri totali e tiri in porta)
    # stat_type="shooting" restituisce un DataFrame con le colonne 'Sh' e 'SoT'
    df_shooting = fbref.read_player_season_stats(stat_type="shooting")
    
    # Scarica le statistiche standard (gol, cartellini, minuti)
    df_standard = fbref.read_player_season_stats(stat_type="standard")
    
    # Uniamo i due DataFrame in base a 'player' e 'team'
    # I nomi delle colonne possono variare, quindi rinominiamo per chiarezza
    df_merged = pd.merge(
        df_standard[['player', 'team', 'Gls', 'Min', 'CrdY']],
        df_shooting[['player', 'team', 'Sh', 'SoT']],
        on=['player', 'team'],
        how='left'
    )
    
    # Rinominiamo le colonne in italiano per renderle leggibili
    df_merged = df_merged.rename(columns={
        'player': 'Giocatore',
        'team': 'Squadra',
        'Gls': 'Gol',
        'Min': 'Minuti',
        'CrdY': 'Cartellini Gialli',
        'Sh': 'Tiri Totali',
        'SoT': 'Tiri in Porta'
    })
    
    # Gestiamo i valori mancanti (es. giocatori senza tiri)
    df_merged = df_merged.fillna(0)
    
    # Calcoliamo le medie per 90 minuti per normalizzare i dati
    # Evitiamo la divisione per zero
    df_merged['Minuti'] = df_merged['Minuti'].replace(0, np.nan)
    df_merged['Tiri/90'] = (df_merged['Tiri Totali'] / df_merged['Minuti'] * 90).fillna(0)
    df_merged['Tiri in Porta/90'] = (df_merged['Tiri in Porta'] / df_merged['Minuti'] * 90).fillna(0)
    df_merged['Gol/90'] = (df_merged['Gol'] / df_merged['Minuti'] * 90).fillna(0)
    df_merged['Cartellini/90'] = (df_merged['Cartellini Gialli'] / df_merged['Minuti'] * 90).fillna(0)
    
    # Teniamo solo le colonne che ci interessano
    colonne_finali = ['Giocatore', 'Squadra', 'Minuti', 'Gol', 'Tiri Totali', 'Tiri in Porta', 'Cartellini Gialli', 'Tiri/90', 'Tiri in Porta/90', 'Gol/90', 'Cartellini/90']
    return df_merged[colonne_finali]

# --- FUNZIONE PER FILTRARE I DATI PER LE SQUADRE SELEZIONATE ---
def filtra_giocatori(df, squadra_casa, squadra_trasferta):
    # Mappatura dei nomi delle squadre (FBref vs football-data.co.uk)
    # Aggiungi qui eventuali altre squadre se i nomi non corrispondono
    mappatura = {
        "Man United": "Manchester United",
        "Man City": "Manchester City",
        "Nott'm Forest": "Nottingham Forest",
        "Sheffield United": "Sheffield United",
        "West Ham": "West Ham United",
        "Newcastle": "Newcastle United",
        "Tottenham": "Tottenham Hotspur",
        "Wolves": "Wolverhampton Wanderers",
        "Brighton": "Brighton & Hove Albion"
    }
    
    # Applichiamo la mappatura ai nomi delle squadre dell'app
    sq_casa = mappatura.get(squadra_casa, squadra_casa)
    sq_trasferta = mappatura.get(squadra_trasferta, squadra_trasferta)
    
    # Filtriamo il DataFrame
    df_filtrato = df[df['Squadra'].isin([sq_casa, sq_trasferta])].copy()
    
    # Se non troviamo nulla, restituiamo None per gestire l'errore
    if df_filtrato.empty:
        return None
    
    # Ordiniamo per Tiri/90 per mostrare i giocatori più pericolosi
    return df_filtrato.sort_values(by='Tiri/90', ascending=False)

# --- INTERFACCIA PRINCIPALE ---
st.title("⚽ Pronostici Calcio Avanzati")
st.write("Seleziona le due squadre per vedere le probabilità calcolate.")

# Addestriamo il modello (come prima)
@st.cache_data
def carica_dati_modello():
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    data = pd.read_csv(url)
    return data.dropna(subset=["FTHG", "FTAG", "HC", "AC", "HY", "AY"])

@st.cache_resource
def allena_modello(data):
    model = pb.models.DixonColesGoalModel(
        goals_home=data["FTHG"],
        goals_away=data["FTAG"],
        teams_home=data["HomeTeam"],
        teams_away=data["AwayTeam"],
    )
    model.fit()
    return model

with st.spinner("Caricamento dati e addestramento del modello in corso..."):
    data = carica_dati_modello()
    model = allena_modello(data)

squadre = sorted(data['HomeTeam'].unique())
col1, col2 = st.columns(2)

with col1:
    casa = st.selectbox("Squadra in Casa", squadre, index=squadre.index("Arsenal") if "Arsenal" in squadre else 0)
with col2:
    trasferta = st.selectbox("Squadra in Trasferta", squadre, index=squadre.index("Chelsea") if "Chelsea" in squadre else 1)

# Inizializziamo la memoria
if "pred" not in st.session_state:
    st.session_state.pred = None
if "match" not in st.session_state:
    st.session_state.match = None

if st.button("Calcola Pronostico", type="primary"):
    if casa == trasferta:
        st.error("Seleziona due squadre diverse!")
    else:
        st.session_state.pred = model.predict(casa, trasferta)
        st.session_state.match = (casa, trasferta)

# Se abbiamo dei risultati, li mostriamo
if st.session_state.pred is not None:
    pred = st.session_state.pred
    casa_match, trasferta_match = st.session_state.match
    
    st.header(f"{casa_match} vs {trasferta_match}")
    
    # ... (Sezioni 1-6 identiche al codice precedente: Mercati Base, Risultati Esatti, HT/FT, Angoli, Cartellini) ...
    # Per non perdere spazio, incolla qui le sezioni 1-6 identiche a prima.
    
    st.divider()

    # --- SEZIONE 7: MERCATI GIOCATORI (STATISTICHE STAGIONALI) ---
    st.subheader("👤 Mercati Giocatori (Statistiche Stagionali)")
    st.write("Clicca il pulsante per caricare le statistiche medie stagionali dei giocatori. I dati vengono da FBref e sono normalizzati per 90 minuti.")
    
    if st.button("Carica Statistiche Giocatori"):
        with st.spinner("Download e elaborazione dati da FBref... (potrebbe volerci un minuto la prima volta)"):
            try:
                df_stats = carica_stats_giocatori()
                df_filtrato = filtra_giocatori(df_stats, casa_match, trasferta_match)
                
                if df_filtrato is None:
                    st.warning(f"Nessun dato trovato per {casa_match} o {trasferta_match}. Controlla la mappatura dei nomi delle squadre nel codice.")
                else:
                    st.success("✅ Dati caricati con successo!")
                    st.dataframe(df_filtrato, use_container_width=True, hide_index=True)
                    
                    # Mostriamo un rapido "top 3" per i tiri
                    st.markdown("#### 🎯 Top 3 per Tiri in Porta / 90 min")
                    top3 = df_filtrato.nlargest(3, 'Tiri in Porta/90')[['Giocatore', 'Squadra', 'Tiri in Porta/90']]
                    st.dataframe(top3, use_container_width=True, hide_index=True)
                    
            except Exception as e:
                st.error(f"Errore durante il download da FBref: {e}")
                st.info("Se l'errore persiste, FBref potrebbe aver temporaneamente bloccato le richieste. Riprova tra qualche minuto.")
