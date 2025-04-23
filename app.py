import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Chargement du modèle ---
with open("modele_football_calibre.pkl", "rb") as f:
    model = pickle.load(f)

# --- Fonction de prédiction ---
def predict_result(features):
    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]
    return prediction, proba

# --- Interface utilisateur ---
st.set_page_config(page_title="Prédiction Match de Foot", layout="centered")
st.title("🌟 Prédiction de résultat de match de football")

st.write("Remplis les caractéristiques ci-dessous pour prédire l'issue d'un match")

# --- Formulaire ---
p_home = st.slider("Probabilité Poisson - Victoire domicile", 0.0, 1.0, 0.4)
p_draw = st.slider("Probabilité Poisson - Nul", 0.0, 1.0, 0.3)
p_away = st.slider("Probabilité Poisson - Victoire extérieure", 0.0, 1.0, 0.3)

diff_form_GF = st.slider("Différentiel de forme offensive (5 derniers matchs)", -3.0, 3.0, 0.0)
diff_form_GA = st.slider("Différentiel de forme défensive (5 derniers matchs)", -3.0, 3.0, 0.0)
diff_odds = st.slider("Différentiel de cotes B365 (dom - ext)", -2.0, 2.0, 0.0)

# --- Prédiction ---
if st.button("🎯 Prédire le résultat"):
    features = {
        "P_home_dc": p_home,
        "P_draw_dc": p_draw,
        "P_away_dc": p_away,
        "diff_avg_form_GF": diff_form_GF,
        "diff_avg_form_GA": diff_form_GA,
        "diff_odds": diff_odds,
    }

    pred, proba = predict_result(features)
    result_map = {0: "Victoire domicile", 1: "Match nul", 2: "Victoire extérieure"}

    st.markdown(f"### 🎉 Résultat prédit : **{result_map[pred]}**")
    st.markdown("### 📊 Probabilités :")
    st.write({result_map[i]: round(prob, 2) for i, prob in enumerate(proba)})
