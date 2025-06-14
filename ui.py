# app/ui.py

import streamlit as st
import pandas as pd
from app.recommender import SteamRecommender

def set_background():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to right, #1c1c1c, #2c2c54);
            color: white;
        }
        .title {
            font-size: 48px;
            color: #ff4b4b;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #f1f1f1;
        }
        .recommendation-box {
            background-color: #333;
            padding: 15px;
            margin: 10px 0;
            border-radius: 12px;
            color: #fff;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        }
        .stButton button {
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            font-size: 16px;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

def launch_ui():
    set_background()

    st.markdown('<div class="title">🎮 Steam Game Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Find games similar to the ones you enjoy!</div>', unsafe_allow_html=True)
    st.write("---")

    # Load and prepare the dataset
    df = pd.read_csv("data/steam-200k.csv", header=None)
    df.columns = ['user_id', 'game', 'behavior', 'value', 'other']

    recommender = SteamRecommender(df)
    recommender.preprocess()

    all_games = df[df['behavior'] == 'play']["game"].dropna().unique()
    selected_game = st.selectbox("🎯 Select a game you like:", sorted(all_games))

    if st.button("🚀 Recommend"):
        recommendations = recommender.recommend_games(selected_game)
        if len(recommendations) > 0:
            st.markdown("## 🎁 You might also like:")
            for game in recommendations:
                st.markdown(f"<div class='recommendation-box'>🎮 {game}</div>", unsafe_allow_html=True)
        else:
            st.error("No recommendations found. Try a different game.")
