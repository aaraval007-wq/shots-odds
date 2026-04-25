import pandas as pd
from datetime import datetime
import numpy as np
import requests 
from io import StringIO
class PreparePastData:
    def __init__(self):
       self.url =  "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

       self.leagues = {
        'premier_league': 'E0',
        'bundesliga':     'D1',
        'la_liga':        'SP1',
        'serie_a':        'I1'}
       
       self.seasons = self._generate_seasons()
       self.raw = self._download_all()

    def _generate_seasons(self) -> list[str]:
        current_year = datetime.now().year
        start_year = current_year - 10
        end_year = current_year + 1

        return [
            f"{str(y)[2:]}{str(y + 1)[2:]}"
            for y in range(start_year, end_year)
        ]

    def _download_season(self, league_code: str, season: str) -> pd.DataFrame | None:
        url = self.url.format(season=season, league=league_code)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text), encoding='latin-1')
                df['season'] = season
                return df
        except requests.RequestException as e:
            print(f"Request error for {url}: {e}")
        return None
    
    def _download_all(self) -> pd.DataFrame:
        all_dfs = []
        for league_name, league_code in self.leagues.items():
            for season in self.seasons:
                df = self._download_season(league_code, season)
                if df is not None:
                    df['league'] = league_name
                    all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()