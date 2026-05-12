from prepare_data import PreparePastData
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import nbinom

class PrepModel:
    def __init__(self, home=['home_adj_shots', 'home_adj_shots_conc', 'prob_h'], 
                       away=['away_adj_shots', 'away_adj_shots_conc', 'prob_a']):
        
        self.home = home
        self.away = away

        self.past_data = PreparePastData()
        self.data = self.past_data.data

        self.X_home = sm.add_constant(self.data[self.home])
        self.X_away = sm.add_constant(self.data[self.away])
        self.y_home = self.data['home_shots']
        self.y_away = self.data['away_shots']

        self.model_home = self._train_model(self.X_home, self.y_home)
        self.model_away = self._train_model(self.X_away, self.y_away)

    def _train_model(self, X, y):
        model = sm.NegativeBinomial(y, X).fit(disp=False)
        return model

    def summary(self):
        print("=== HOME MODEL ===")
        print(self.model_home.summary())
        print("\n=== AWAY MODEL ===")
        print(self.model_away.summary())

        


class PredictModel:
    def __init__(self):
        self.models = PrepModel()
        self.model_home = self.models.model_home
        self.model_away = self.models.model_away
        self.home_features = self.models.home
        self.away_features = self.models.away
        self.alpha_home = self.models.model_home.params['alpha']
        self.alpha_away = self.models.model_away.params['alpha']
        self.data = self.models.past_data.future_data
        self.future_data = self.predict()

    def predict(self):
        X_home = sm.add_constant(self.data[self.home_features], has_constant='add')
        X_away = sm.add_constant(self.data[self.away_features], has_constant='add')

        future_data = self.data.drop(columns=['home_shots', 'away_shots'], errors='ignore')
        future_data['mu_home'] = self.model_home.predict(X_home)
        future_data['mu_away'] = self.model_away.predict(X_away)

        return future_data

