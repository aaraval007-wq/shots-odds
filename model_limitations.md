1. opp_rolling_shots_conc has no venue adjustment — a team on a run of away games looks like a weaker defence than they are, and vice versa

2. opp_rolling_shots_conc has no opponent adjustment — a team who've faced strong attackers recently looks like a weaker defence than they are

3. Early-season data is sparse — rolling windows of 5 with min 3 means the first few gameweeks of every season are dropped entirely, reducing training data

4. No cross-season continuity — a team's last 3 games of season N are ignored when computing rolling features for their first games of season N+1, despite being genuinely informative

5. Odds sourced from B365 only for past data, averaged across bookmakers for future data — this inconsistency means prob_h/prob_d/prob_a are not computed on a like-for-like basis across the full dataset

6. Football-data.co.uk only covers 4 leagues for past data — Ligue 1 future matches have odds but no past shot history to build rolling features from, so they'll always be dropped by the min_periods filter

7. Raw shots used rather than xG — shots are a noisier signal than expected goals; a long-range speculative effort counts the same as a clear-cut chance

8. No red card or injury data — a team reduced to 10 men generates far fewer shots, but this is invisible to the model
9. No match importance or rotation signal — cup finals, dead rubber league games, and heavily rotated XIs all distort shot patterns in ways the rolling average cannot detect

10. Rolling window is fixed at 5 games — this is a hyperparameter that has never been tuned; 3 or 7 might produce better signal depending on the league