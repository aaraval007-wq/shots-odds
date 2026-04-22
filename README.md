# Shots Odds

This document contains a project to predict and gamble on the number of shots in a match and by the individual team in any give premier league, la liga, serie A, bundesliga or ligue 1 match.
ing, rolling shots a team has taken in the last 5 matches adjusted to the team they played against, rolling shots conceded by the opposing team adjusted to the team they played against.

In the current state of the project:
- I am building a data layer, which automatically takes in the match data and creates our features for past and future matches
- I am building the model layer, which takes all past matches to build the model and then uses it to predict future matches

output will be a probability distribution function of shots by the teams playing
