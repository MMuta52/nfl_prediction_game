import pandas as pd
import numpy as np
import util
from strategy.Strategy import Strategy

HFA = 65.0     # Home field advantage is worth 65 Elo points
K = 20.0       # The speed at which Elo ratings change
REVERSION_FACTOR = 1/3.0 # Between seasons, a team retains 2/3 of its previous season's rating
REVERSION_BASE = 1550.0

class FiveThirtyEightBaseStrategy(Strategy):
  """
  Follows the baseline methodology outlined here: https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
  """
  def __init__(self, seasons):
    super().__init__('base538', seasons)
  
  def prepare_data(self):
    df = util.load_data(self.seasons)
    return df

  def offseason_reversion(self, game):
    self.elo[game.team1] = REVERSION_BASE * REVERSION_FACTOR + self.elo[game.team1] * (1-REVERSION_FACTOR)
    self.elo[game.team2] = REVERSION_BASE * REVERSION_FACTOR + self.elo[game.team2] * (1-REVERSION_FACTOR)
  
  def update_elo(self, game, elo_diff, prediction):
    # Margin of victory is used as a K multiplier
    if game.result1 == 0.5:
      winner_elo_diff = 1.0
    elif game.result1 == 1:
      winner_elo_diff = elo_diff
    else:
      winner_elo_diff = -elo_diff

    winner_point_diff = max(abs(game.score1 - game.score2), 1)
    mult = np.log(winner_point_diff + 1.0) * (2.2 / (winner_elo_diff * 0.001 + 2.2))

    # Elo shift based on K and the margin of victory multiplier
    shift = (K * mult) * (game.result1 - prediction)

    # Apply shift
    self.elo[game.team1] += shift
    self.elo[game.team2] -= shift

  def simulate(self, schedule_df):
    self.elo = self.initialize_elos(schedule_df)
    my_pred = []
    my_elo1 = []
    my_elo2 = []

    for game in schedule_df.itertuples():
      # Apply offseason reversion (except in first week of simulation)
      if game.week == 1 and game.season != schedule_df['season'].min():
        self.offseason_reversion(game)

      # Elo difference includes home field advantage
      elo_diff = self.elo[game.team1] - self.elo[game.team2] + (0 if game.neutral == 1 else HFA)

      # Elo prediction method
      prediction = 1.0 / ((10.0 ** (-elo_diff/400.0)) + 1.0)

      # Store elos and prediction
      my_elo1.append(self.elo[game.team1])
      my_elo2.append(self.elo[game.team2])
      my_pred.append(prediction)

      # Update Elo ratings
      self.update_elo(game, elo_diff, prediction)

    return self.format_output(schedule_df, my_elo1, my_elo2, my_pred)

