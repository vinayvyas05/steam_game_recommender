import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# -------------------- Recommender Class --------------------
class SteamRecommender:
    def __init__(self, df):
        self.df = df
        self.user_encoder = LabelEncoder()
        self.game_encoder = LabelEncoder()
        self.pivot_table = None
        self.similarity = None

    def preprocess(self):
        self.df.columns = ['user_id', 'game', 'behavior', 'value', 'other']
        self.df = self.df[self.df['behavior'] == 'play']
        self.df = self.df[self.df['value'] > 0]

        self.df["user"] = self.user_encoder.fit_transform(self.df["user_id"])
        self.df["game_enc"] = self.game_encoder.fit_transform(self.df["game"])
        
        self.pivot_table = self.df.pivot_table(
            index="user", columns="game_enc", values="value", fill_value=0
        )
        self.similarity = cosine_similarity(self.pivot_table.T)

    def recommend_games(self, game_title, top_n=5):
        try:
            game_idx = self.game_encoder.transform([game_title])[0]
        except:
            return []
        similarity_scores = list(enumerate(self.similarity[game_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        similar_game_indices = [i[0] for i in similarity_scores]
        return self.game_encoder.inverse_transform(similar_game_indices)

# -------------------- UI Logic --------------------
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

    df = pd.read_csv("steam-200k.csv", header=None)
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

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    launch_ui()
