import streamlit as st
import pandas as pd
import penaltyblog as pb
import numpy as np
from scipy.stats import poisson
import requests
from datetime import datetime, timedelta

# Impostazioni della pagina web
st.set_page_config(page_title="Calcio Betting Model", page_icon="⚽", layout="wide")

# --- BARRA LATERALE (SIDEBAR) ---
st.sidebar.header("⚙️ Impostazioni API")
api_key = st.sidebar.text_input("API Key (API-Football)", type="password", help="Inserisci la chiave gratuita di API-Football.")
st.sidebar.info("Se non hai la chiave, l'app funzionerà lo stesso ma i mercati giocatori non saranno disponibili.")

# --- DIZIONARIO ID SQUADRE PREMIER LEAGUE ---
TEAM_IDS = {
    "Arsenal": 42, "Aston Villa": 66, "Bournemouth": 35, "Brentford": 55,
    "Brighton": 51, "Chelsea": 49, "Crystal Palace": 52, "Everton": 45,
    "Fulham": 36, "Liverpool": 40, "Luton": 1359, "Man City": 50,
    "Man United": 33, "Newcastle": 34, "Nott'm Forest": 65,
    "Sheffield United": 62, "Tottenham": 47, "West Ham": 48, "Wolves": 39,
    "Burnley": 44,
}

@st.cache_data
def carica_dati():
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    data = pd.read_csv(url)
    data = data.dropna(subset=["FTHG", "FTAG", "HC", "AC", "HY", "AY"])
    return data

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

@st.cache_data(show_spinner=False)
def get_player_stats(api_key, team_home, team_away):
    headers = {
        'x-apisports-key': api_key,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    # 1. Otteniamo gli ID (dal dizionario o tramite ricerca)
    home_id = TEAM_IDS.get(team_home)
    away_id = TEAM_IDS.get(team_away)
    
    if not home_id:
        r = requests.get(f"https://v3.football.api-sports.io/teams?search={team_home}", headers=headers).json()
        if not r.get('response'):
            return None, f"ID non trovato per {team_home}. Risposta API: {r.get('errors')}"
        home_id = r['response'][0]['team']['id']
        
    if not away_id:
        r = requests.get(f"https://v3.football.api-sports.io/teams?search={team_away}", headers=headers).json()
        if not r.get('response'):
            return None, f"ID non trovato per {team_away}. Risposta API: {r.get('errors')}"
        away_id = r['response'][0]['team']['id']
    
    # 2. Troviamo l'ultima partita tra le due squadre usando un intervallo di date
    # Calcoliamo le date: da 6 mesi fa a oggi
    oggi = datetime.now().strftime('%Y-%m-%d')
    sei_mesi_fa = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # Chiamata headtohead con from e to (parametri accettati dal piano gratuito)
    url_h2h = f"https://v3.football.api-sports.io/fixtures/headtohead?h2h={home_id}-{away_id}&from={sei_mesi_fa}&to={oggi}"
    r_fixture = requests.get(url_h2h, headers=headers).json()
    
    if not r_fixture.get('response'):
        return None, f"Nessuna partita trovata tra {team_home} e {team_away} negli ultimi 6 mesi. Risposta API: {r_fixture.get('errors')}"
    
    # Ordiniamo i risultati per data (dal più recente) e prendiamo il primo
    partite = r_fixture['response']
    partite.sort(key=lambda x: x['fixture']['date'], reverse=True)
    fixture_id = partite[0]['fixture']['id']
    fixture_date = partite[0]['fixture']['date']
    
    # 3. Otteniamo le statistiche dei giocatori per quella partita
    r_stats = requests.get(f"https://v3.football.api-sports.io/fixtures/players?fixture={fixture_id}", headers=headers).json()
    
    if not r_stats.get('response'):
        return None, f"Statistiche non disponibili per la partita del {fixture_date[:10]}. Risposta API: {r_stats.get('errors')}"
    
    # 4. Estraiamo i dati
    players_data = []
    for team_data in r_stats['response']:
        for player in team_data['players']:
            stats = player['statistics'][0]
            
            # Gestione dei valori None
            minuti = stats['games']['minutes'] or 0
            if minuti == 0:
                continue
                
            tiri_totali = stats['shots']['total'] or 0
            tiri_in_porta = stats['shots']['on'] or 0
            gol = stats['goals']['total'] or 0
            cartellini = stats['cards']['yellow'] or 0
            
            players_data.append({
                'Giocatore': player['player']['name'],
                'Squadra': team_data['team']['name'],
                'Minuti': minuti,
                'Tiri Totali': tiri_totali,
                'Tiri in Porta': tiri_in_porta,
                'Gol Segnati': gol,
                'Cartellini Gialli': cartellini
            })
            
    if not players_data:
        return None, "Nessun giocatore con minuti giocati trovato (la partita potrebbe non essere ancora stata giocata)."
        
    df = pd.DataFrame(players_data)
    return df, fixture_date

# --- INTERFACCIA PRINCIPALE ---
st.title("⚽ Pronostici Calcio Avanzati")
st.write("Seleziona le due squadre per vedere le probabilità calcolate.")

with st.spinner("Caricamento dati e addestramento del modello in corso..."):
    data = carica_dati()
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
    
    # ... (Sezioni 1-6: Mercati Base, Risultati Esatti, HT/FT, Angoli, Cartellini) ...
    # Incolla qui le sezioni 1-6 identiche al codice precedente per non perdere spazio.
    # (Puoi copiarle dal messaggio precedente, sono invariate)
    
    st.divider()

    # --- SEZIONE 7: MERCATI GIOCATORI ---
    st.subheader("👤 Mercati Giocatori (Dati Reali da API-Football)")
    st.write("Clicca il pulsante per caricare i dati reali dei giocatori dell'ultima partita tra queste due squadre. (Consuma 3 richieste API).")
    
    if st.button("Carica Statistiche Giocatori"):
        if not api_key:
            st.warning("⚠️ Inserisci la tua API Key nella barra laterale sinistra.")
        else:
            with st.spinner("Interrogazione API-Football in corso..."):
                df_players, info = get_player_stats(api_key, casa_match, trasferta_match)
                
                if df_players is None:
                    st.error(f"Errore: {info}")
                else:
                    st.success(f"✅ Dati caricati con successo! (Riferiti alla partita del {info[:10]})")
                    st.dataframe(df_players, use_container_width=True, hide_index=True)
