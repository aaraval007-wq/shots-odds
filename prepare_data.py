import pandas as pd
from datetime import datetime
import numpy as np
import requests
from io import StringIO
import json
from pathlib import Path


class PreparePastData:
    def __init__(self, patches_path: str = 'manual_odds_patches.json',
                 odds_api_key: str = 'd88a85887c730ce9a8c6575e7ba532b4'):
        self.patches_path = Path(patches_path)
        self.odds_api_key = odds_api_key
        self.url = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

        self.leagues = {
            'premier_league': 'E0',
            'bundesliga':     'D1',
            'la_liga':        'SP1',
            'serie_a':        'I1',
            'ligue_1':        'F1',
        }

        self.odds_api_league_keys = {
            'premier_league': 'soccer_epl',
            'bundesliga':     'soccer_germany_bundesliga',
            'la_liga':        'soccer_spain_la_liga',
            'serie_a':        'soccer_italy_serie_a',
            'ligue_1':        'soccer_france_ligue_one',
        }

        self.seasons = self._generate_seasons()
        self.raw = self._download_all()
        self.data = self._clean()
        self._apply_patches()
        self._create_adj_shots()
        self.all_data = self._combine()

    # ------------------------------------------------------------------ #
    #  PAST DATA — unchanged methods                                       #
    # ------------------------------------------------------------------ #

    def _generate_seasons(self) -> list[str]:
        current_year = datetime.now().year
        start_year = current_year - 10
        end_year = current_year + 1
        # football-data.co.uk season codes look like "1516" (2015-16), "2324" (2023-24)
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

    def _clean(self) -> pd.DataFrame:
        odds_cols = ['B365H', 'B365D', 'B365A']
        COLS = ['Date', 'HomeTeam', 'AwayTeam', 'HS', 'AS'] + odds_cols + ['league', 'season']

        df = self.raw[COLS].copy()
        df.columns = ['date', 'home_team', 'away_team', 'home_shots', 'away_shots',
                      'odds_h', 'odds_d', 'odds_a', 'league', 'season']

        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')

        before = len(df)
        df = df.dropna(subset=['home_team', 'away_team', 'home_shots', 'away_shots'])
        dropped = before - len(df)
        if dropped > 0:
            print(f"[PreparePastData] Dropped {dropped} rows with missing team or shot data.")

        df['prob_h'] = 1 / df['odds_h']
        df['prob_d'] = 1 / df['odds_d']
        df['prob_a'] = 1 / df['odds_a']

        # Divide by the sum to strip the bookmaker's overround (vig), giving true implied probs
        total = df['prob_h'] + df['prob_d'] + df['prob_a']
        df['prob_h'] = df['prob_h'] / total
        df['prob_d'] = df['prob_d'] / total
        df['prob_a'] = df['prob_a'] / total

        df['has_odds'] = df[['prob_h', 'prob_d', 'prob_a']].notna().all(axis=1)

        missing = df[~df['has_odds']][['date', 'home_team', 'away_team', 'league', 'season']]
        if not missing.empty:
            print(f"\n[PreparePastData] {len(missing)} rows are missing odds:")
            print(missing.to_string(index=False))
            print("\nUse patch_odds() to manually supply odds for any of these.\n")

        return df

    def _apply_patches(self) -> None:
        if not self.patches_path.exists():
            return
        with open(self.patches_path, 'r') as f:
            patches = json.load(f)
        applied = 0
        for patch in patches:
            mask = (
                (self.data['home_team'] == patch['home_team']) &
                (self.data['away_team'] == patch['away_team']) &
                (self.data['date'] == pd.Timestamp(patch['date']))
            )
            if mask.any():
                self.data.loc[mask, 'odds_h'] = patch['odds_h']
                self.data.loc[mask, 'odds_d'] = patch['odds_d']
                self.data.loc[mask, 'odds_a'] = patch['odds_a']
                total = (1/patch['odds_h']) + (1/patch['odds_d']) + (1/patch['odds_a'])
                self.data.loc[mask, 'prob_h'] = (1/patch['odds_h']) / total
                self.data.loc[mask, 'prob_d'] = (1/patch['odds_d']) / total
                self.data.loc[mask, 'prob_a'] = (1/patch['odds_a']) / total
                self.data.loc[mask, 'has_odds'] = True
                applied += 1
        if applied > 0:
            print(f"[PreparePastData] Re-applied {applied} manual patch(es) from {self.patches_path}.")

    def patch_odds(self, home_team: str, away_team: str, date: str,
                   odds_h: float, odds_d: float, odds_a: float) -> None:
        patches = []
        if self.patches_path.exists():
            with open(self.patches_path, 'r') as f:
                patches = json.load(f)
        already_exists = any(
            p['home_team'] == home_team and
            p['away_team'] == away_team and
            p['date'] == date
            for p in patches
        )
        if already_exists:
            print(f"[PreparePastData] Patch for {home_team} vs {away_team} on {date} already exists. Skipping.")
            return
        patches.append({
            'home_team': home_team, 'away_team': away_team, 'date': date,
            'odds_h': odds_h, 'odds_d': odds_d, 'odds_a': odds_a
        })
        with open(self.patches_path, 'w') as f:
            json.dump(patches, f, indent=2)
        self._apply_patches()
        print(f"[PreparePastData] Patch saved and applied for {home_team} vs {away_team} on {date}.")

    def get_match(self, home_team: str, away_team: str, date: str) -> pd.DataFrame:
        mask = (
            (self.data['home_team'] == home_team) &
            (self.data['away_team'] == away_team) &
            (self.data['date'] == pd.Timestamp(date))
        )
        result = self.data.loc[mask]
        if result.empty:
            print(f"[PreparePastData] No match found for {home_team} vs {away_team} on {date}.")
        return result

    def _create_adj_shots(self) -> None:
        data = self.data

        home_df = data[['date', 'season', 'league', 'home_team', 'away_team',
                         'home_shots', 'away_shots']].copy()
        home_df = home_df.rename(columns={
            'home_team': 'team', 'away_team': 'opponent',
            'home_shots': 'shots_scored', 'away_shots': 'shots_conceded'
        })
        home_df['venue'] = 'home'

        away_df = data[['date', 'season', 'league', 'away_team', 'home_team',
                         'away_shots', 'home_shots']].copy()
        away_df = away_df.rename(columns={
            'away_team': 'team', 'home_team': 'opponent',
            'away_shots': 'shots_scored', 'home_shots': 'shots_conceded'
        })
        away_df['venue'] = 'away'

        long_df = pd.concat([home_df, away_df], ignore_index=True)
        long_df = long_df.sort_values(['team', 'season', 'date']).reset_index(drop=True)

        def expanding_league_avg(group):
            # shift(1) ensures each row only sees games that have already been played (no lookahead)
            group = group.copy()
            group['expanding_avg_home_shots'] = group['home_shots'].expanding().mean().shift(1)
            group['expanding_avg_away_shots'] = group['away_shots'].expanding().mean().shift(1)
            # Midpoint between home/away league averages — the "neutral-venue" shot baseline
            group['expanding_neutral'] = (group['expanding_avg_home_shots'] + group['expanding_avg_away_shots']) / 2
            # Per-team average shots conceded in a neutral context (total / 2)
            group['expanding_avg_shots_conc'] = (group['home_shots'] + group['away_shots']).expanding().mean().shift(1) / 2
            return group

        league_season_stats = (data.sort_values('date')
                               .groupby(['league', 'season'], group_keys=False)
                               .apply(expanding_league_avg))

        league_season_stats = league_season_stats[['date', 'league', 'season',
                                                    'expanding_avg_home_shots',
                                                    'expanding_avg_away_shots',
                                                    'expanding_neutral',
                                                    'expanding_avg_shots_conc']]

        def rolling_shots_conc(group):
            group = group.copy()
            group['opp_rolling_shots_conc'] = (group['shots_conceded']
                                               .shift(1)
                                               .rolling(window=5, min_periods=3)
                                               .mean())
            return group

        long_df = (long_df.sort_values(['team', 'season', 'date'])
                   .groupby(['team', 'season'], group_keys=False)
                   .apply(rolling_shots_conc))

        # Multiple matches share the same date; .last() keeps the most up-to-date expanding average
        league_season_stats_dedup = (league_season_stats
                                     .sort_values('date')
                                     .groupby(['date', 'league', 'season'])
                                     .last()
                                     .reset_index())

        long_df = long_df.merge(league_season_stats_dedup, on=['date', 'league', 'season'], how='left')

        # Remove venue bias: subtract the home premium for home teams, add it back for away teams,
        # so both sides are expressed on the same neutral-venue scale
        long_df['neutral_shots'] = long_df.apply(
            lambda row: row['shots_scored'] - (row['expanding_avg_home_shots'] - row['expanding_neutral'])
            if row['venue'] == 'home'
            else row['shots_scored'] + (row['expanding_neutral'] - row['expanding_avg_away_shots']),
            axis=1
        )

        opp_stats = long_df[['team', 'date', 'season', 'opp_rolling_shots_conc']].copy()
        opp_stats = opp_stats.rename(columns={
            'team': 'opponent',
            'opp_rolling_shots_conc': 'opp_def_strength'
        })

        long_df = long_df.merge(opp_stats, on=['opponent', 'date', 'season'], how='left')

        # Scale neutral shots by how tough the opponent's defence is relative to league average:
        # ratio > 1 means opponent concedes more than average (weak defence) → deflate shots,
        # ratio < 1 means opponent concedes less (strong defence) → inflate shots
        long_df['adj_shots'] = (long_df['neutral_shots'] *
                                (long_df['expanding_avg_shots_conc'] / long_df['opp_def_strength']))

        def rolling_adj_shots(group):
            group = group.copy()
            group['rolling_adj_shots'] = (group['adj_shots']
                                          .shift(1)
                                          .rolling(window=5, min_periods=3)
                                          .mean())
            return group

        long_df = (long_df.sort_values(['team', 'season', 'date'])
                   .groupby(['team', 'season'], group_keys=False)
                   .apply(rolling_adj_shots))

        home_features = long_df[long_df['venue'] == 'home'][['team', 'date', 'season',
                                                               'rolling_adj_shots',
                                                               'opp_rolling_shots_conc']].copy()
        home_features.columns = ['home_team', 'date', 'season', 'home_adj_shots', 'home_adj_shots_conc']

        away_features = long_df[long_df['venue'] == 'away'][['team', 'date', 'season',
                                                               'rolling_adj_shots',
                                                               'opp_rolling_shots_conc']].copy()
        away_features.columns = ['away_team', 'date', 'season', 'away_adj_shots', 'away_adj_shots_conc']

        data = data.merge(home_features, on=['date', 'home_team', 'season'], how='left')
        data = data.merge(away_features, on=['date', 'away_team', 'season'], how='left')

        data = data.dropna(subset=['home_adj_shots', 'home_adj_shots_conc',
                                   'away_adj_shots', 'away_adj_shots_conc'])

        self.data = data
        self._long_df = long_df  # cache for reuse in future adj shots computation

    # ------------------------------------------------------------------ #
    #  FUTURE DATA                                                         #
    # ------------------------------------------------------------------ #

    def _fetch_future_odds(self) -> pd.DataFrame:
        rows = []
        for league_name, sport_key in self.odds_api_league_keys.items():
            try:
                url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
                params = {
                    "apiKey":     self.odds_api_key,
                    "regions":    "uk",
                    "markets":    "h2h",
                    "oddsFormat": "decimal",
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                raw = resp.json()

                for event in raw:
                    home, away = event["home_team"], event["away_team"]
                    h_odds, d_odds, a_odds = [], [], []

                    for bookmaker in event["bookmakers"]:
                        for market in bookmaker.get("markets", []):
                            if market["key"] != "h2h":
                                continue
                            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                            if home in outcomes and away in outcomes and "Draw" in outcomes:
                                h_odds.append(outcomes[home])
                                d_odds.append(outcomes["Draw"])
                                a_odds.append(outcomes[away])

                    if not h_odds:
                        continue

                    rows.append({
                        "league":    league_name,
                        "date":      pd.to_datetime(event["commence_time"]).tz_convert("Europe/London").tz_localize(None),
                        "home_team": home,
                        "away_team": away,
                        "odds_h":    round(sum(h_odds) / len(h_odds), 3),
                        "odds_d":    round(sum(d_odds) / len(d_odds), 3),
                        "odds_a":    round(sum(a_odds) / len(a_odds), 3),
                    })

                print(f"  [Odds API] {league_name}: {len([r for r in rows if r['league'] == league_name])} matches fetched")

            except requests.HTTPError as e:
                print(f"  [Odds API] {league_name}: HTTP error — {e}")
            except Exception as e:
                print(f"  [Odds API] {league_name}: unexpected error — {e}")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

        # Normalise probabilities
        df['prob_h'] = 1 / df['odds_h']
        df['prob_d'] = 1 / df['odds_d']
        df['prob_a'] = 1 / df['odds_a']
        total = df['prob_h'] + df['prob_d'] + df['prob_a']
        df['prob_h'] = df['prob_h'] / total
        df['prob_d'] = df['prob_d'] / total
        df['prob_a'] = df['prob_a'] / total
        df['has_odds'] = True

        return df

    def _create_future_adj_shots(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling_adj_shots and opp_rolling_shots_conc for future matches
        using only past data. Drops matches where either team has fewer than
        3 games in their rolling window.
        """
        # Work out current season — most recent season in past data
        current_season = self.data['season'].max()

        # Pull the cached long_df from past data — we'll extend it with future rows
        past_long = self._long_df.copy()

        # Build future long rows — one home, one away per match, shots unknown
        future_home = future_df[['date', 'league', 'home_team', 'away_team']].copy()
        future_home = future_home.rename(columns={'home_team': 'team', 'away_team': 'opponent'})
        future_home['venue'] = 'home'
        future_home['season'] = current_season
        future_home['shots_scored'] = np.nan
        future_home['shots_conceded'] = np.nan

        future_away = future_df[['date', 'league', 'away_team', 'home_team']].copy()
        future_away = future_away.rename(columns={'away_team': 'team', 'home_team': 'opponent'})
        future_away['venue'] = 'away'
        future_away['season'] = current_season
        future_away['shots_scored'] = np.nan
        future_away['shots_conceded'] = np.nan

        future_long = pd.concat([future_home, future_away], ignore_index=True)

        # Stack past + future long rows, sort by team/season/date
        combined = pd.concat([past_long, future_long], ignore_index=True)
        combined = combined.sort_values(['team', 'season', 'date']).reset_index(drop=True)

        # Recompute rolling_adj_shots across the combined df
        # For future rows adj_shots is NaN so they don't pollute the window,
        # but their rolling_adj_shots will correctly look back into past rows
        def rolling_adj_shots(group):
            group = group.copy()
            group['rolling_adj_shots'] = (group['adj_shots']
                                          .shift(1)
                                          .rolling(window=5, min_periods=3)
                                          .mean())
            return group

        # adj_shots is NaN on future rows so they don't corrupt the rolling window,
        # but their rolling_adj_shots still picks up the last N past values correctly
        combined = (combined.sort_values(['team', 'season', 'date'])
                    .groupby(['team', 'season'], group_keys=False)
                    .apply(rolling_adj_shots))

        # Recompute opp_rolling_shots_conc the same way
        def rolling_shots_conc(group):
            group = group.copy()
            group['opp_rolling_shots_conc'] = (group['shots_conceded']
                                               .shift(1)
                                               .rolling(window=5, min_periods=3)
                                               .mean())
            return group

        combined = (combined.sort_values(['team', 'season', 'date'])
                    .groupby(['team', 'season'], group_keys=False)
                    .apply(rolling_shots_conc))

        # Extract only future rows
        future_only = combined[combined['shots_scored'].isna()].copy()

        # Split back to home/away and rejoin onto future_df
        home_features = (future_only[future_only['venue'] == 'home']
                         [['team', 'date', 'season', 'rolling_adj_shots', 'opp_rolling_shots_conc']]
                         .copy())
        home_features.columns = ['home_team', 'date', 'season', 'home_adj_shots', 'home_adj_shots_conc']

        away_features = (future_only[future_only['venue'] == 'away']
                         [['team', 'date', 'season', 'rolling_adj_shots', 'opp_rolling_shots_conc']]
                         .copy())
        away_features.columns = ['away_team', 'date', 'season', 'away_adj_shots', 'away_adj_shots_conc']

        future_df = future_df.copy()
        future_df['season'] = current_season
        future_df = future_df.merge(home_features, on=['date', 'home_team', 'season'], how='left')
        future_df = future_df.merge(away_features, on=['date', 'away_team', 'season'], how='left')

        # Drop matches where either team lacks enough history for a rolling window
        before = len(future_df)
        future_df = future_df.dropna(subset=['home_adj_shots', 'home_adj_shots_conc','away_adj_shots', 'away_adj_shots_conc'])
        dropped = before - len(future_df)
        if dropped > 0:
            print(f"[PreparePastData] Dropped {dropped} future match(es) with insufficient rolling history.")

        return future_df.reset_index(drop=True)

    def _combine(self) -> pd.DataFrame:
        """
        Returns a single df of past + future matches.
        Past rows have home_shots, away_shots and all rolling features.
        Future rows have predictor variables only (shots are NaN).
        """
        future_odds = self._fetch_future_odds()

        if future_odds.empty:
            print("[PreparePastData] No future odds fetched — returning past data only.")
            return self.data.copy()

        future_with_features = self._create_future_adj_shots(future_odds)

        # Align columns — future rows will have NaN for shot actuals
        future_with_features['home_shots'] = np.nan
        future_with_features['away_shots'] = np.nan
        future_with_features['is_future'] = True

        self.future_data = future_with_features  # cache for inspection

        past = self.data.copy()
        past['is_future'] = False

        combined = pd.concat([past, future_with_features], ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)

        return combined