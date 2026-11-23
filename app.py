import streamlit as st
import pandas as pd
import penaltyblog as pb

# Impostazioni della pagina web
st.set_page_config(page_title="Calcio Betting Model", page_icon="⚽")

# Funzione per scaricare i dati (la salviamo in memoria per non riscaricarli ogni volta)
@st.cache_data
def carica_dati():
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    data = pd.read_csv(url)
    data = data.dropna(subset=["FTHG", "FTAG"])
    return data

# Funzione per addestrare il modello
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

# Titolo dell'app
st.title("⚽ Pronostici Calcio (Premier League)")
st.write("Seleziona le due squadre per vedere le probabilità calcolate dal modello Dixon-Coles.")

# Scarichiamo i dati e alleniamo il modello
with st.spinner("Caricamento dati e addestramento del modello in corso..."):
    data = carica_dati()
    model = allena_modello(data)

# Creiamo i menu a tendina
squadre = sorted(data['HomeTeam'].unique())
col1, col2 = st.columns(2)

with col1:
    casa = st.selectbox("Squadra in Casa", squadre, index=squadre.index("Arsenal") if "Arsenal" in squadre else 0)
with col2:
    trasferta = st.selectbox("Squadra in Trasferta", squadre, index=squadre.index("Chelsea") if "Chelsea" in squadre else 1)

# Bottone per calcolare
if st.button("Calcola Pronostico", type="primary"):
    if casa == trasferta:
        st.error("Seleziona due squadre diverse!")
    else:
        pred = model.predict(casa, trasferta)
        
        st.subheader(f"{casa} vs {trasferta}")
        
        # Mostriamo le probabilità in tre colonne
        c1, c2, c3 = st.columns(3)
        c1.metric("Vittoria Casa (1)", f"{pred.home_win:.1%}")
        c2.metric("Pareggio (X)", f"{pred.draw:.1%}")
        c3.metric("Vittoria Trasferta (2)", f"{pred.away_win:.1%}")
        
        st.write("---")
        
        # Altri mercati
        c4, c5 = st.columns(2)
        c4.metric("Over 2.5", f"{pred.total_goals('over', 2.5):.1%}")
        c5.metric("Gol (BTTS Sì)", f"{pred.btts_yes:.1%}")
