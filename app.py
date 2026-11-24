import streamlit as st
import pandas as pd
import penaltyblog as pb
import numpy as np
from scipy.stats import poisson

# Impostazioni della pagina web
st.set_page_config(page_title="Calcio Betting Model", page_icon="⚽", layout="wide")

# Funzione per scaricare i dati
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
st.title("⚽ Pronostici Calcio Avanzati (Premier League)")
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
        
        st.header(f"{casa} vs {trasferta}")
        
        # --- SEZIONE 1: MERCATI BASE ---
        st.subheader("Mercati Principali")
        c1, c2, c3 = st.columns(3)
        c1.metric("Vittoria Casa (1)", f"{pred.home_win:.1%}")
        c2.metric("Pareggio (X)", f"{pred.draw:.1%}")
        c3.metric("Vittoria Trasferta (2)", f"{pred.away_win:.1%}")
        
        c4, c5 = st.columns(2)
        c4.metric("Over 2.5", f"{pred.total_goals('over', 2.5):.1%}")
        c5.metric("Gol (BTTS Sì)", f"{pred.btts_yes:.1%}")
        
        st.divider()
        
        # --- SEZIONE 2: RISULTATI ESATTI ---
        st.subheader("🎯 Risultati Esatti (Top 8)")
        st.write("Le probabilità dei punteggi più probabili per le tue multiple.")
        
        scores = []
        for h in range(5):
            for a in range(5):
                prob = pred.exact_score(h, a)
                scores.append({"Punteggio": f"{h}-{a}", "Probabilità": prob})
        
        df_scores = pd.DataFrame(scores)
        df_scores = df_scores.sort_values(by="Probabilità", ascending=False).head(8)
        
        cols = st.columns(4)
        for i, row in df_scores.reset_index(drop=True).iterrows():
            with cols[i % 4]:
                st.metric(label=row['Punteggio'], value=f"{row['Probabilità']:.1%}")
        
        st.divider()
        
        # --- SEZIONE 3: ESITO TEMPO/FINALE (HT/FT) ---
        st.subheader("⏱️ Esito Tempo / Finale (HT/FT)")
        st.write("Probabilità combinata di come finisce il 1° Tempo e la partita. I colori più scuri indicano le probabilità più alte.")
        
        # Stima dei gol: assumiamo che il 45% dei gol avvenga nel primo tempo
        lambda_home_ht = pred.home_goal_expectation * 0.45
        lambda_away_ht = pred.away_goal_expectation * 0.45
        
        prob_ht = []
        for h in range(5):
            for a in range(5):
                prob_ht.append({"h": h, "a": a, "p": poisson.pmf(h, lambda_home_ht) * poisson.pmf(a, lambda_away_ht)})
        
        combinazioni = {
            "1/1": 0.0, "1/X": 0.0, "1/2": 0.0,
            "X/1": 0.0, "X/X": 0.0, "X/2": 0.0,
            "2/1": 0.0, "2/X": 0.0, "2/2": 0.0
        }
        
        def get_esito(h, a):
            if h > a: return "1"
            elif h == a: return "X"
            else: return "2"
        
        lambda_home_st = pred.home_goal_expectation * 0.55
        lambda_away_st = pred.away_goal_expectation * 0.55
        
        for ht in prob_ht:
            if ht['p'] < 0.001: continue
            esito_ht = get_esito(ht['h'], ht['a'])
            
            for st_h in range(5):
                for st_a in range(5):
                    p_st = poisson.pmf(st_h, lambda_home_st) * poisson.pmf(st_a, lambda_away_st)
                    ft_h = ht['h'] + st_h
                    ft_a = ht['a'] + st_a
                    esito_ft = get_esito(ft_h, ft_a)
                    combinazione = f"{esito_ht}/{esito_ft}"
                    combinazioni[combinazione] += ht['p'] * p_st
                    
        # --- CREAZIONE DELLA TABELLA BELLA E COLORATA ---
        data_matrix = []
        for e_ht in ["1", "X", "2"]:
            row = []
            for e_ft in ["1", "X", "2"]:
                row.append(combinazioni[f"{e_ht}/{e_ft}"])
            data_matrix.append(row)
            
        df_htft = pd.DataFrame(
            data_matrix,
            index=["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
            columns=["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"]
        )
        df_htft.index.name = "1° Tempo ⬇️ / Finale ➡️"
        
        styled_htft = df_htft.style.format("{:.1%}").background_gradient(cmap="Blues", axis=None)
        
        st.dataframe(styled_htft, use_container_width=True)
