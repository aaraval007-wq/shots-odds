import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import pandas as pd
from model import PredictModel


class ShotPredictor:
    def __init__(self):
        self.predictor = PredictModel()
        self.data = self.predictor.future_data

    # ------------------------------------------------------------------ #
    #  PUBLIC                                                              #
    # ------------------------------------------------------------------ #

    def plot(self, date: str, team: str) -> None:
        """
        Plot the NegBin shot probability distribution for a given team
        in a given match.

        Parameters
        ----------
        date : str
            Match date in 'YYYY-MM-DD' format.
        team : str
            Team name exactly as it appears in the data (football-data naming).
        """
        row, venue = self._find_match(date, team)
        if row is None:
            return

        mu    = row[f'mu_{venue}']
        alpha = self._get_alpha(venue)

        x, pmf = self._negbin_pmf(mu, alpha)

        self._plot_distribution(x, pmf, team, date, mu, alpha)

    def matches_on_date(self, date: str) -> pd.DataFrame:
        """
        Return all matches scheduled on a given date.

        Parameters
        ----------
        date : str
            Date in 'YYYY-MM-DD' format.
        """
        import datetime

        mask = self.data['date'].dt.date == datetime.date.fromisoformat(date)
        matches = self.data[mask][['date', 'league', 'home_team', 'away_team']].copy()

        if matches.empty:
            print(f"[ShotDistribution] No matches found on {date}.")
        else:
            print(matches.to_string(index=False))

        return matches
    
    def betway_best_bet(self, date: str, team: str, odds: dict[int, float]) -> pd.DataFrame:
        """
        Compare Betway 'x+ shots' market odds against NegBin model probabilities
        using half-Kelly criterion to find the strongest bet.

        Parameters
        ----------
        date : str
            Match date in 'YYYY-MM-DD' format.
        team : str
            Team name as it appears in the data.
        odds : dict[int, float]
            Dictionary of {threshold: decimal_odds} e.g. {10: 1.1, 11: 1.25, ...}
            Each entry represents 'threshold+ shots'.
        """
        row, venue = self._find_match(date, team)
        if row is None:
            return

        mu    = row[f'mu_{venue}']
        alpha = self._get_alpha(venue)

        # Always start from 0 so no threshold can fall outside the computed range
        n       = 1.0 / alpha
        p       = n / (n + mu)
        sd      = np.sqrt(mu + alpha * mu ** 2)
        x_max   = int(np.ceil(mu + 3 * sd))
        x       = np.arange(0, x_max + 1)
        pmf     = nbinom.pmf(x, n, p)

        rows = []
        for threshold, decimal_odds in sorted(odds.items()):
            # P(shots >= threshold) — strictly cumulative from threshold upward
            model_prob  = pmf[x >= threshold].sum()
            market_prob = 1 / decimal_odds
            edge        = model_prob - market_prob

            b      = decimal_odds - 1
            kelly  = 0.5 * (b * model_prob - (1 - model_prob)) / b
            kelly  = max(kelly, 0)

            rows.append({
                'threshold':   f'{threshold}+',
                'odds':        decimal_odds,
                'market_prob': round(market_prob, 4),
                'model_prob':  round(model_prob, 4),
                'edge':        round(edge, 4),
                'half_kelly':  round(kelly, 4),
            })

        results = (pd.DataFrame(rows)
                .sort_values('half_kelly', ascending=False)
                .reset_index(drop=True))

        positive_ev = results[results['edge'] > 0]
        best        = results.iloc[0]

        print(f"\n{'='*55}")
        print(f"  {team}  |  {date}")
        print(f"  μ = {mu:.2f}  |  α = {alpha:.4f}")
        print(f"{'='*55}")
        print(results.to_string(index=False))
        print(f"{'='*55}")

        if positive_ev.empty:
            print("  No positive EV found across all given odds — no bet recommended.")
        elif best['half_kelly'] == 0:
            print("  Positive EV exists but Kelly stake is zero — no bet recommended.")
        else:
            print(f"  Best bet : {best['threshold']} shots @ {best['odds']}")
            print(f"  Edge     : {best['edge']:.2%}")
            print(f"  Stake    : {best['half_kelly']:.2%} of bankroll")

        print(f"{'='*55}\n")

        return results

    # ------------------------------------------------------------------ #
    #  INTERNAL                                                            #
    # ------------------------------------------------------------------ #

    def _find_match(self, date: str, team: str) -> tuple:
        """
        Locate the row for the given date and team.
        Returns (row, venue) where venue is 'home' or 'away', or (None, None).
        """
        target_date = date  # kept as string for .dt.date comparison

        mask_date = self.data['date'].dt.date == __import__('datetime').date.fromisoformat(date)
        day_df    = self.data[mask_date]

        if day_df.empty:
            print(f"[ShotDistribution] No matches found on {date}.")
            return None, None

        home_mask = day_df['home_team'] == team
        away_mask = day_df['away_team'] == team

        if home_mask.any():
            return day_df[home_mask].iloc[0], 'home'
        elif away_mask.any():
            return day_df[away_mask].iloc[0], 'away'
        else:
            available = sorted(
                set(day_df['home_team'].tolist() + day_df['away_team'].tolist())
            )
            print(f"[ShotDistribution] '{team}' not found on {date}.")
            print(f"  Available teams on that date: {available}")
            return None, None

    def _get_alpha(self, venue: str) -> float:
        """Return the fitted alpha (dispersion) for home or away model."""
        if venue == 'home':
            return self.predictor.alpha_home
        return self.predictor.alpha_away

    def _negbin_pmf(self, mu: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute NegBin PMF over a ± 3 SD range clipped to non-negative integers.

        Statsmodels NegBin parameterisation: Var = mu + alpha * mu^2
        scipy.stats.nbinom uses (n, p) parameterisation:
            n = 1 / alpha
            p = n / (n + mu)
        """
        n = 1.0 / alpha
        p = n / (n + mu)

        sd    = np.sqrt(mu + alpha * mu ** 2)
        x_min = max(0, int(np.floor(mu - 3 * sd)))
        x_max = int(np.ceil(mu + 3 * sd))

        x   = np.arange(x_min, x_max + 1)
        pmf = nbinom.pmf(x, n, p)

        return x, pmf

    def _plot_distribution(self, x: np.ndarray, pmf: np.ndarray,
                           team: str, date: str,
                           mu: float, alpha: float) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))

        bars = ax.bar(x, pmf, color='steelblue', edgecolor='white', linewidth=0.6, alpha=0.85)

        # Highlight the most likely outcome
        peak_idx = np.argmax(pmf)
        bars[peak_idx].set_color('tomato')
        bars[peak_idx].set_alpha(1.0)

        # Mean line
        ax.axvline(mu, color='navy', linestyle='--', linewidth=1.4, label=f'Mean (μ = {mu:.2f})')

        ax.set_xlabel('Shots', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'{team} — Shot Distribution\n{date}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)

        # Annotate peak bar
        ax.annotate(
            f'Mode: {x[peak_idx]}  ({pmf[peak_idx]:.1%})',
            xy=(x[peak_idx], pmf[peak_idx]),
            xytext=(x[peak_idx] + 0.5, pmf[peak_idx] + 0.005),
            fontsize=9,
            color='tomato'
        )

        sd = np.sqrt(mu + alpha * mu ** 2)
        ax.text(
            0.97, 0.95,
            f'α = {alpha:.4f}\nσ = {sd:.2f}',
            transform=ax.transAxes,
            fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='grey', alpha=0.8)
        )

        ax.set_xticks(x)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.tight_layout()
        plt.show()