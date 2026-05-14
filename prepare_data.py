import pandas as pd
from datetime import datetime
import numpy as np
import requests
from io import StringIO
import json
from pathlib import Path
from rapidfuzz import process, fuzz


class PreparePastData:
    def __init__(self,
                 patches_path: str = 'manual_odds_patches.json',
                 name_map_path: str = 'team_name_map.json',
                 odds_api_key: str = 'd88a85887c730ce9a8c6575e7ba532b4'):
        self.patches_path = Path(patches_path)
        self.name_map_path = Path(name_map_path)
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
    #  SEASONS                                                             #
    # ------------------------------------------------------------------ #

    def _generate_seasons(self) -> list[str]:
        current_year = datetime.now().year
        start_year = current_year - 10
        end_year = current_year + 1
        return [
            f"{str(y)[2:]}{str(y + 1)[2:]}"
            for y in range(start_year, end_year)
        ]

    # ------------------------------------------------------------------ #
    #  DOWNLOAD                                                            #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  CLEAN                                                               #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  ODDS PATCHES                                                        #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  TEAM NAME MAPPING                                                   #
    # ------------------------------------------------------------------ #

    def _load_name_map(self) -> dict[str, str]:
        """Load confirmed odds-API name -> football-data name mappings from disk."""
        if not self.name_map_path.exists():
            return {}
        with open(self.name_map_path, 'r') as f:
            return json.load(f)

    def _save_name_map(self, name_map: dict[str, str]) -> None:
        with open(self.name_map_path, 'w') as f:
            json.dump(name_map, f, indent=2, sort_keys=True)

    def _apply_name_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Translate odds-API team names to their football-data equivalents.

        Names that are already exact matches to a football-data name pass
        through unchanged. Names in the confirmed map are translated. Any
        remaining unknown names cause that match to be dropped with a warning
        prompting the user to run review_team_names().
        """
        name_map = self._load_name_map()
        fd_names = set(self.data['home_team']).union(set(self.data['away_team']))

        df = df.copy()

        def translate(name: str) -> str | None:
            if name in fd_names:
                return name
            if name in name_map:
                return name_map[name]
            return None

        df['home_team'] = df['home_team'].map(translate)
        df['away_team'] = df['away_team'].map(translate)

        unknown_mask = df['home_team'].isna() | df['away_team'].isna()
        unknown_count = unknown_mask.sum()

        if unknown_count > 0:
            print(
                f"\n[PreparePastData] {unknown_count} future match(es) dropped — "
                f"team name(s) not in name map.\n"
                f"  Run pp.review_team_names() to resolve missing mappings.\n"
            )
            df = df[~unknown_mask]

        return df.reset_index(drop=True)

    def review_team_names(self) -> None:
        """
        Interactive workflow to resolve unmatched odds-API team names.

        For each odds-API name not already in the confirmed map or the
        football-data name list, shows the 2 closest football-data candidates
        and prompts you to pick one or enter a manual override. Confirmed
        mappings are saved to team_name_map.json immediately so you never
        have to resolve the same name twice.

        Only names that genuinely differ between sources need entries —
        exact matches are skipped automatically.
        """
        name_map = self._load_name_map()
        fd_names = sorted(set(self.data['home_team']).union(set(self.data['away_team'])))

        print("[review_team_names] Fetching current future odds to collect team names...")
        future_odds = self._fetch_future_odds()
        if future_odds.empty:
            print("[review_team_names] No future odds available — nothing to review.")
            return

        odds_api_names = sorted(
            set(future_odds['home_team']).union(set(future_odds['away_team']))
        )

        unresolved = [
            name for name in odds_api_names
            if name not in fd_names and name not in name_map
        ]

        if not unresolved:
            print("[review_team_names] All team names are already mapped. Nothing to do.")
            return

        print(f"\n[review_team_names] {len(unresolved)} unresolved name(s) to review.")
        print("  Enter the number of your chosen match, 'm' to type manually, or 's' to skip.\n")

        newly_confirmed = 0

        for odds_name in unresolved:
            candidates = process.extract(
                odds_name,
                fd_names,
                scorer=fuzz.WRatio,
                limit=2
            )

            print(f"  Odds-API name : '{odds_name}'")
            for i, (candidate, score, _) in enumerate(candidates, start=1):
                print(f"    [{i}] '{candidate}'  (score: {score:.0f})")
            print(f"    [m] Enter manually")
            print(f"    [s] Skip for now")

            while True:
                choice = input("  Your choice: ").strip().lower()

                if choice in ('s', ''):
                    print(f"  Skipped '{odds_name}' — match will be dropped until resolved.\n")
                    break

                elif choice == 'm':
                    manual = input("  Type the correct football-data name: ").strip()
                    if manual in fd_names:
                        name_map[odds_name] = manual
                        self._save_name_map(name_map)
                        print(f"  Saved: '{odds_name}' -> '{manual}'\n")
                        newly_confirmed += 1
                        break
                    else:
                        print(f"  '{manual}' not found in football-data names. Try again.")

                elif choice in ('1', '2'):
                    idx = int(choice) - 1
                    chosen = candidates[idx][0]
                    name_map[odds_name] = chosen
                    self._save_name_map(name_map)
                    print(f"  Saved: '{odds_name}' -> '{chosen}'\n")
                    newly_confirmed += 1
                    break

                else:
                    print("  Invalid input. Enter 1, 2, m, or s.")

        print(f"\n[review_team_names] Done. {newly_confirmed} new mapping(s) saved to {self.name_map_path}.")

    # ------------------------------------------------------------------ #
    #  ADJUSTED SHOTS                                                      #
    # ------------------------------------------------------------------ #

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
            group = group.copy()
            group['expanding_avg_home_shots'] = group['home_shots'].expanding().mean().shift(1)
            group['expanding_avg_away_shots'] = group['away_shots'].expanding().mean().shift(1)
            group['expanding_neutral'] = (group['expanding_avg_home_shots'] + group['expanding_avg_away_shots']) / 2
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

        league_season_stats_dedup = (league_season_stats
                                     .sort_values('date')
                                     .groupby(['date', 'league', 'season'])
                                     .last()
                                     .reset_index())

        long_df = long_df.merge(league_season_stats_dedup, on=['date', 'league', 'season'], how='left')

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
        self._long_df = long_df

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
        current_season = self.data['season'].max()
        past_long = self._long_df.copy()
        past_long['_is_future'] = False

        future_home = future_df[['date', 'league', 'home_team', 'away_team']].copy()
        future_home = future_home.rename(columns={'home_team': 'team', 'away_team': 'opponent'})
        future_home['venue'] = 'home'
        future_home['season'] = current_season
        future_home['shots_scored'] = np.nan
        future_home['shots_conceded'] = np.nan
        future_home['_is_future'] = True

        future_away = future_df[['date', 'league', 'away_team', 'home_team']].copy()
        future_away = future_away.rename(columns={'away_team': 'team', 'home_team': 'opponent'})
        future_away['venue'] = 'away'
        future_away['season'] = current_season
        future_away['shots_scored'] = np.nan
        future_away['shots_conceded'] = np.nan
        future_away['_is_future'] = True

        future_long = pd.concat([future_home, future_away], ignore_index=True)
        combined = pd.concat([past_long, future_long], ignore_index=True)
        combined = combined.sort_values(['team', 'season', 'date']).reset_index(drop=True)

        def rolling_adj_shots(group):
            group = group.copy()
            group['rolling_adj_shots'] = (group['adj_shots']
                                          .shift(1)
                                          .rolling(window=5, min_periods=3)
                                          .mean())
            return group

        def rolling_shots_conc(group):
            group = group.copy()
            group['opp_rolling_shots_conc'] = (group['shots_conceded']
                                               .shift(1)
                                               .rolling(window=5, min_periods=3)
                                               .mean())
            return group

        combined = (combined.sort_values(['team', 'season', 'date'])
                    .groupby(['team', 'season'], group_keys=False)
                    .apply(rolling_adj_shots))

        combined = (combined.sort_values(['team', 'season', 'date'])
                    .groupby(['team', 'season'], group_keys=False)
                    .apply(rolling_shots_conc))

        future_only = combined[combined['_is_future']].copy()

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

        before = len(future_df)
        future_df = future_df.dropna(subset=['home_adj_shots', 'home_adj_shots_conc',
                                             'away_adj_shots', 'away_adj_shots_conc'])
        dropped = before - len(future_df)
        if dropped > 0:
            print(f"[PreparePastData] Dropped {dropped} future match(es) with insufficient rolling history.")

        return future_df.reset_index(drop=True)

    def _combine(self) -> pd.DataFrame:
        future_odds = self._fetch_future_odds()

        if future_odds.empty:
            print("[PreparePastData] No future odds fetched — returning past data only.")
            return self.data.copy()

        # Translate odds-API names to football-data names before any merges
        future_odds = self._apply_name_map(future_odds)

        if future_odds.empty:
            print("[PreparePastData] No future matches remaining after name mapping — returning past data only.")
            return self.data.copy()

        future_with_features = self._create_future_adj_shots(future_odds)

        future_with_features['home_shots'] = np.nan
        future_with_features['away_shots'] = np.nan
        future_with_features['is_future'] = True

        self.future_data = future_with_features

        past = self.data.copy()
        past['is_future'] = False

        combined = pd.concat([past, future_with_features], ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)

        return combined