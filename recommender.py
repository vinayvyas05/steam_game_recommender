# app/recommender.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

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
