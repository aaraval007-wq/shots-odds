import pandas as pd
from datetime import datetime
import numpy as np
import requests
from io import StringIO
import json
from pathlib import Path


class PreparePastData:
    def __init__(self, patches_path: str = 'manual_odds_patches.json'):
        # patches_path must be set FIRST — other methods (_apply_patches) reference
        # it, and if __init__ crashes mid-way, those methods still need it available
        self.patches_path = Path(patches_path)

        # Template URL for football-data.co.uk — {season} and {league} are filled
        # in dynamically by _download_season
        self.url = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

        # League codes as used by football-data.co.uk
        self.leagues = {
            'premier_league': 'E0',
            'bundesliga':     'D1',
            'la_liga':        'SP1',
            'serie_a':        'I1'
        }

        # --- Pipeline runs in order on instantiation ---
        self.seasons = self._generate_seasons()   # work out which seasons to pull
        self.raw = self._download_all()           # download everything into one df
        self.data = self._clean()                 # select, rename, validate columns
        self._apply_patches()                     # restore any manually saved odds

        self._create_adj_shots()                  # create columns for rolling adjusted shots taken and conceded

    def _generate_seasons(self) -> list[str]:
        current_year = datetime.now().year

        # Always pull 10 years of history — start_year is 10 years ago
        start_year = current_year - 10

        # end_year is next calendar year, so we attempt to pull the upcoming season
        # (e.g. in 2026 this gives us 2627). If it doesn't exist yet on the site,
        # _download_season handles the 404 gracefully
        end_year = current_year + 1

        # Format as 2-digit year pairs: 2016 → "1617", 2026 → "2627"
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
                df['season'] = season  # tag each row with its season before merging
                return df
            # Non-200 (e.g. 404 for a future season) — return None silently
        except requests.RequestException as e:
            # Network errors (timeout, DNS failure etc.) are caught here
            # so one bad request doesn't abort the whole download
            print(f"Request error for {url}: {e}")
        return None

    def _download_all(self) -> pd.DataFrame:
        all_dfs = []
        for league_name, league_code in self.leagues.items():
            for season in self.seasons:
                df = self._download_season(league_code, season)
                if df is not None:
                    df['league'] = league_name  # tag each row with its league
                    all_dfs.append(df)

        # Return empty DataFrame rather than crashing if nothing downloaded
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def _clean(self) -> pd.DataFrame:
        # Prefer Max odds (best available market price) but fall back to B365
        # if Max columns are absent or entirely empty. We check notna().any() rather
        # than just column existence because after pd.concat, a column that only
        # existed in some CSVs will still appear in self.raw — just full of NaNs
        '''if 'MaxH' in self.raw.columns and self.raw['MaxH'].notna().any():
            odds_cols = ['MaxH', 'MaxD', 'MaxA']
        else:'''
        odds_cols = ['B365H', 'B365D', 'B365A']

        COLS = ['Date', 'HomeTeam', 'AwayTeam', 'HS', 'AS'] + odds_cols + ['league', 'season']

        df = self.raw[COLS].copy()
        df.columns = ['date', 'home_team', 'away_team', 'home_shots', 'away_shots',
                    'odds_h', 'odds_d', 'odds_a', 'league', 'season']

        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')

        # DROP rows missing team names or shot counts
        before = len(df)
        df = df.dropna(subset=['home_team', 'away_team', 'home_shots', 'away_shots'])
        dropped = before - len(df)
        if dropped > 0:
            print(f"[PreparePastData] Dropped {dropped} rows with missing team or shot data.")

        # Convert raw odds to normalised probabilities
        df['prob_h'] = 1 / df['odds_h']
        df['prob_d'] = 1 / df['odds_d']
        df['prob_a'] = 1 / df['odds_a']

        total = df['prob_h'] + df['prob_d'] + df['prob_a']
        df['prob_h'] = df['prob_h'] / total
        df['prob_d'] = df['prob_d'] / total
        df['prob_a'] = df['prob_a'] / total

        # KEEP rows with missing odds but flag them
        df['has_odds'] = df[['prob_h', 'prob_d', 'prob_a']].notna().all(axis=1)

        missing = df[~df['has_odds']][['date', 'home_team', 'away_team', 'league', 'season']]
        if not missing.empty:
            print(f"\n[PreparePastData] {len(missing)} rows are missing odds:")
            print(missing.to_string(index=False))
            print("\nUse patch_odds() to manually supply odds for any of these.\n")

        return df

    def _apply_patches(self) -> None:
        # If no patch file exists yet, nothing to do
        if not self.patches_path.exists():
            return

        with open(self.patches_path, 'r') as f:
            patches = json.load(f)

        applied = 0
        for patch in patches:
            # Match rows by home team, away team, and exact date
            mask = (
                (self.data['home_team'] == patch['home_team']) &
                (self.data['away_team'] == patch['away_team']) &
                (self.data['date'] == pd.Timestamp(patch['date']))
            )
            if mask.any():
                # Write raw odds back in, then recompute normalised probabilities
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
        # Load existing patches from disk so we can check for duplicates
        patches = []
        if self.patches_path.exists():
            with open(self.patches_path, 'r') as f:
                patches = json.load(f)

        # Avoid writing the same match twice — if you re-run the notebook and call
        # patch_odds again for the same match, this silently skips it
        already_exists = any(
            p['home_team'] == home_team and
            p['away_team'] == away_team and
            p['date'] == date
            for p in patches
        )

        if already_exists:
            print(f"[PreparePastData] Patch for {home_team} vs {away_team} on {date} already exists. Skipping.")
            return

        # Save to disk first — this is what persists the patch across runs
        patches.append({
            'home_team': home_team,
            'away_team': away_team,
            'date': date,
            'odds_h': odds_h,
            'odds_d': odds_d,
            'odds_a': odds_a
        })

        with open(self.patches_path, 'w') as f:
            json.dump(patches, f, indent=2)

        # Also apply immediately to self.data so the current session reflects the fix
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

        # 1
        home_df = data[['date', 'season', 'league', 'home_team', 'away_team', 'home_shots', 'away_shots']].copy()
        home_df = home_df.rename(columns={
            'home_team': 'team',
            'away_team': 'opponent',
            'home_shots': 'shots_scored',
            'away_shots': 'shots_conceded'
        })
        home_df['venue'] = 'home'

        away_df = data[['date', 'season', 'league', 'away_team', 'home_team', 'away_shots', 'home_shots']].copy()
        away_df = away_df.rename(columns={
            'away_team': 'team',
            'home_team': 'opponent',
            'away_shots': 'shots_scored',
            'home_shots': 'shots_conceded'
        })
        away_df['venue'] = 'away'

        long_df = pd.concat([home_df, away_df], ignore_index=True)
        long_df = long_df.sort_values(['team', 'season', 'date']).reset_index(drop=True)


        # 2

        # For each league-season, compute expanding average shots conceded up to each date
        # We use all teams' shots conceded in that league-season up to that point

        league_season_expanding = (data.sort_values('date')
                                .groupby(['league', 'season']))

        def expanding_league_avg(group):
            # Total shots in each game for that league/season up to that point
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


        # 3

        # For each team, compute rolling avg shots conceded over last 5 games, same season only, min 3 games

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


        # 4

        # Take the last expanding value for each date/league/season
        # (after all games on that date have been processed)
        league_season_stats_dedup = (league_season_stats
                                    .sort_values('date')
                                    .groupby(['date', 'league', 'season'])
                                    .last()
                                    .reset_index())

        # Now merge
        long_df = long_df.merge(
            league_season_stats_dedup,
            on=['date', 'league', 'season'],
            how='left'
        )

        # 5

        long_df['neutral_shots'] = long_df.apply(
            lambda row: row['shots_scored'] - (row['expanding_avg_home_shots'] - row['expanding_neutral'])
            if row['venue'] == 'home'
            else row['shots_scored'] + (row['expanding_neutral'] - row['expanding_avg_away_shots']),
            axis=1
        )

        # 6

        # We need the opponent's rolling shots conceded at the time of each match
        # This is the opp_rolling_shots_conc from the opponent's perspective

        opp_stats = long_df[['team', 'date', 'season', 'opp_rolling_shots_conc']].copy()
        opp_stats = opp_stats.rename(columns={
            'team': 'opponent',
            'opp_rolling_shots_conc': 'opp_def_strength'
        })

        long_df = long_df.merge(
            opp_stats,
            on=['opponent', 'date', 'season'],
            how='left'
        )

        # 7

        # Apply opponent strength scaling
        long_df['adj_shots'] = (long_df['neutral_shots'] *
                                (long_df['expanding_avg_shots_conc'] / long_df['opp_def_strength']))

        # Now compute rolling mean of adj_shots over last 5 games, same season, min 3 games
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
        
        # 8

        # Split long_df back into home and away
        home_features = long_df[long_df['venue'] == 'home'][['team', 'date', 'season',
                                                            'rolling_adj_shots',
                                                            'opp_rolling_shots_conc']].copy()
        home_features.columns = ['home_team', 'date', 'season',
                                'home_adj_shots',
                                'home_adj_shots_conc']

        away_features = long_df[long_df['venue'] == 'away'][['team', 'date', 'season',
                                                            'rolling_adj_shots',
                                                            'opp_rolling_shots_conc']].copy()
        away_features.columns = ['away_team', 'date', 'season',
                                'away_adj_shots',
                                'away_adj_shots_conc']

        # Merge onto original dataframe
        data = data.merge(home_features, on=['date', 'home_team', 'season'], how='left')
        data = data.merge(away_features, on=['date', 'away_team', 'season'], how='left')

        # 9

        data = data.dropna(subset=['home_adj_shots', 'home_adj_shots_conc',
                                    'away_adj_shots', 'away_adj_shots_conc'])
        
        self.data = data

            
