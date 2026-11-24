import streamlit as st
import pandas as pd
import penaltyblog as pb
import numpy as np
from scipy.stats import poisson
import requests

# Impostazioni della pagina web
st.set_page_config(page_title="Calcio Betting Model", page_icon="⚽", layout="wide")

# Funzione per scaricare i dati
@st.cache_data
def carica_dati():
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    data = pd.read_csv(url)
    data = data.dropna(subset=["FTHG", "FTAG", "HC", "AC", "HY", "AY"])
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
st.title("⚽ Pronostici Calcio Avanzati")
st.write("Seleziona le due squadre per vedere le probabilità calcolate.")

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
        st.subheader("1️⃣ Pronostico di Base")
        c1, c2, c3 = st.columns(3)
        c1.metric("Vittoria Casa (1)", f"{pred.home_win:.1%}")
        c2.metric("Pareggio (X)", f"{pred.draw:.1%}")
        c3.metric("Vittoria Trasferta (2)", f"{pred.away_win:.1%}")
        
        c4, c5 = st.columns(2)
        c4.metric("Over 2.5", f"{pred.total_goals('over', 2.5):.1%}")
        c5.metric("Gol (BTTS Sì)", f"{pred.btts_yes:.1%}")
        
        st.divider()
        
        # --- SEZIONE 2: RISULTATI ESATTI FINALI ---
        st.subheader("🎯 Risultati Esatti più probabili (Fine Partita)")
        
        scores = []
        for h in range(5):
            for a in range(5):
                prob = pred.exact_score(h, a)
                scores.append({"Punteggio": f"{h}-{a}", "Probabilità": prob, "Quota Minima": 1/prob})
        
        df_scores = pd.DataFrame(scores)
        df_scores = df_scores.sort_values(by="Probabilità", ascending=False).head(5)
        
        styled_scores = df_scores.style.format({
            'Probabilità': '{:.1%}',
            'Quota Minima': '{:.2f}'
        }).background_gradient(cmap="Greens", subset=['Probabilità'])
        
        st.dataframe(styled_scores, use_container_width=True, hide_index=True)

        p_top3 = df_scores['Probabilità'].head(3).sum()
        st.info(f"💡 **Consiglio Multipla:** Se giochi i primi 3 risultati esatti insieme ({df_scores['Punteggio'].iloc[0]}, {df_scores['Punteggio'].iloc[1]}, {df_scores['Punteggio'].iloc[2]}), la probabilità totale è del **{p_top3:.1%}** (Quota minima: {1/p_top3:.2f}).")

        st.divider()

        # --- SEZIONE 3: RISULTATI ESATTI PRIMO TEMPO ---
        st.subheader("⏱️ Risultati Esatti più probabili (Solo 1° Tempo)")
        
        lambda_home_ht = pred.home_goal_expectation * 0.45
        lambda_away_ht = pred.away_goal_expectation * 0.45
        
        scores_ht = []
        for h in range(3):
            for a in range(3):
                prob = poisson.pmf(h, lambda_home_ht) * poisson.pmf(a, lambda_away_ht)
                scores_ht.append({"Punteggio 1°T": f"{h}-{a}", "Probabilità": prob, "Quota Minima": 1/prob})
                
        df_scores_ht = pd.DataFrame(scores_ht)
        df_scores_ht = df_scores_ht.sort_values(by="Probabilità", ascending=False).head(4)
        
        styled_scores_ht = df_scores_ht.style.format({
            'Probabilità': '{:.1%}',
            'Quota Minima': '{:.2f}'
        }).background_gradient(cmap="Blues", subset=['Probabilità'])
        
        st.dataframe(styled_scores_ht, use_container_width=True, hide_index=True)

        st.divider()

        # --- SEZIONE 4: COMBO TEMPO/FINALE (HT/FT) ---
        st.subheader("🔄 Combo: Esito 1° Tempo / Esito Finale")
        
        lambda_home_st = pred.home_goal_expectation * 0.55
        lambda_away_st = pred.away_goal_expectation * 0.55
        
        combinazioni = []
        nomi_esiti = {"1": "Casa (1)", "X": "Pareggio (X)", "2": "Trasferta (2)"}
        
        for h_ht in range(5):
            for a_ht in range(5):
                p_ht = poisson.pmf(h_ht, lambda_home_ht) * poisson.pmf(a_ht, lambda_away_ht)
                if p_ht < 0.0001: continue
                
                esito_ht = "1" if h_ht > a_ht else ("X" if h_ht == a_ht else "2")
                
                for h_st in range(5):
                    for a_st in range(5):
                        p_st = poisson.pmf(h_st, lambda_home_st) * poisson.pmf(a_st, lambda_away_st)
                        if p_st < 0.0001: continue
                        
                        ft_h = h_ht + h_st
                        ft_a = a_ht + a_st
                        esito_ft = "1" if ft_h > ft_a else ("X" if ft_h == ft_a else "2")
                        
                        combo_name = f"{esito_ht}/{esito_ft}"
                        prob_totale = p_ht * p_st
                        combinazioni.append({"Combo (1°T / Finale)": combo_name, "Probabilità": prob_totale})
        
        df_combo = pd.DataFrame(combinazioni).groupby("Combo (1°T / Finale)")["Probabilità"].sum().reset_index()
        df_combo = df_combo.sort_values(by="Probabilità", ascending=False)
        df_combo["Quota Minima"] = 1 / df_combo["Probabilità"]
        
        df_combo["Significato"] = df_combo["Combo (1°T / Finale)"].apply(
            lambda x: f"1°T: {nomi_esiti[x.split('/')[0]]} -> Finale: {nomi_esiti[x.split('/')[1]]}"
        )
        
        styled_combo = df_combo[["Combo (1°T / Finale)", "Significato", "Probabilità", "Quota Minima"]].style.format({
            'Probabilità': '{:.1%}',
            'Quota Minima': '{:.2f}'
        }).background_gradient(cmap="Oranges", subset=['Probabilità'])
        
        st.dataframe(styled_combo, use_container_width=True, hide_index=True)
        
        st.divider()

        # --- SEZIONE 5: ANGOLI (CORNERS) ---
        st.subheader("🚩 Mercato Angoli (Corners)")
        
        avg_home_corners = data[data['HomeTeam'] == casa]['HC'].mean()
        avg_away_corners = data[data['AwayTeam'] == trasferta]['AC'].mean()
        lambda_corners = avg_home_corners + avg_away_corners
        
        st.write(f"**Stima Angoli Totali:** {lambda_corners:.1f} (Media {casa} in casa: {avg_home_corners:.1f} | Media {trasferta} in trasferta: {avg_away_corners:.1f})")
        
        linee_corners = [8.5, 9.5, 10.5, 11.5]
        cols_corners = st.columns(len(linee_corners))
        
        for i, linea in enumerate(linee_corners):
            p_over = 1 - poisson.cdf(np.floor(linea), lambda_corners)
            with cols_corners[i]:
                st.metric(f"Over {linea}", f"{p_over:.1%}", help=f"Quota minima: {1/p_over:.2f}")

        st.divider()

        # --- SEZIONE 6: CARTELLINI (CARDS) ---
        st.subheader("🟨 Mercato Cartellini (Ammonizioni)")
        
        avg_home_cards = data[data['HomeTeam'] == casa]['HY'].mean()
        avg_away_cards = data[data['AwayTeam'] == trasferta]['AY'].mean()
        lambda_cards = avg_home_cards + avg_away_cards
        
        st.write(f"**Stima Cartellini Totali:** {lambda_cards:.1f} (Media {casa} in casa: {avg_home_cards:.1f} | Media {trasferta} in trasferta: {avg_away_cards:.1f})")
        
        linee_cards = [2.5, 3.5, 4.5, 5.5]
        cols_cards = st.columns(len(linee_cards))
        
        for i, linea in enumerate(linee_cards):
            p_over = 1 - poisson.cdf(np.floor(linea), lambda_cards)
            with cols_cards[i]:
                st.metric(f"Over {linea}", f"{p_over:.1%}", help=f"Quota minima: {1/p_over:.2f}")

        st.divider()

        # --- SEZIONE 7: MERCATI GIOCATORI (API-FOOTBALL) ---
        st.subheader("👤 Mercati Giocatori (Marcatori e Tiri)")
        st.write("Clicca il pulsante per caricare i dati dei giocatori da API-Football. **Attenzione:** consuma 1 richiesta del tuo piano gratuito (100 al giorno).")
        
        # Inserisci qui la tua chiave API (per il test) o usa st.secrets
        # Per semplicità, la mettiamo in un campo di testo nascosto
        api_key = st.text_input("Inserisci la tua API Key di API-Football", type="password")
        
        if st.button("Carica Statistiche Giocatori"):
            if not api_key:
                st.error("Inserisci la tua API Key per procedere.")
            else:
                with st.spinner("Interrogazione API-Football in corso..."):
                    try:
                        # 1. Troviamo l'ID della partita (fixture) tramite le squadre
                        # Questo è un esempio, la ricerca precisa richiede la data esatta
                        # Per semplicità, cerchiamo la partita più recente tra le due squadre
                        # In un'app reale, dovresti usare un selettore di data.
                        # Usiamo un endpoint di ricerca generico (fittizio per l'esempio)
                        # In API-Football, si usa /fixtures?team=...&team=...&last=1
                        
                        # Per questo esempio, simuliamo una risposta per mostrare la struttura.
                        # In un caso reale, dovresti fare:
                        # headers = {'x-rapidapi-key': api_key, 'x-rapidapi-host': 'v3.football.api-sports.io'}
                        # url_fixture = f"https://v3.football.api-sports.io/fixtures?team={casa_id}&team={trasferta_id}&last=1"
                        # response = requests.get(url_fixture, headers=headers).json()
                        # fixture_id = response['response'][0]['fixture']['id']
                        
                        # Poi:
                        # url_stats = f"https://v3.football.api-sports.io/fixtures/players?fixture={fixture_id}"
                        # stats = requests.get(url_stats, headers=headers).json()
                        
                        # Per ora, mostriamo una tabella di esempio per illustrare l'output
                        # Inseriremo qui la logica reale nel prossimo passaggio, se vuoi.
                        
                        st.info("Funzionalità di esempio. La logica per trovare l'ID della partita e chiamare l'API verrà implementata nel prossimo step, in base al tuo feedback.")
                        
                        # Tabella di esempio (fittizia) per mostrare come apparirà
                        df_players_example = pd.DataFrame({
                            'Giocatore': ['Saka', 'Martinelli', 'Odegaard', 'Havertz', 'Palmer', 'Jackson', 'Sterling'],
                            'Squadra': ['Arsenal', 'Arsenal', 'Arsenal', 'Arsenal', 'Chelsea', 'Chelsea', 'Chelsea'],
                            'Tiri Totali': [3.5, 2.8, 2.1, 1.9, 3.2, 2.5, 2.0],
                            'Tiri in Porta': [1.2, 0.9, 0.7, 0.5, 1.1, 0.8, 0.6],
                            'Gol Segnati': [0.5, 0.4, 0.2, 0.2, 0.4, 0.3, 0.2],
                            'Cartellini': [0.1, 0.1, 0.05, 0.1, 0.15, 0.2, 0.1]
                        })
                        st.dataframe(df_players_example, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Errore durante la chiamata API: {e}")
