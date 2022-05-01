import pandas as pd
import numpy as np

HFA = 65.0     # Home field advantage is worth 65 Elo points
K = 20.0       # The speed at which Elo ratings change
REVERSION_FACTOR = 1/3.0 # Between seasons, a team retains 2/3 of its previous season's rating
REVERSION_BASE = 1550.0

class Strategy:
  """
  A Strategy has a simulate() function that takes a schedule_df and returns a DataFrame of predictions
  """
  def __init__(self, name):
    self.name = name
    self.elo = {}

  def initialize_elos(self, schedule_df):
    return pd.concat([schedule_df[['date','team1','elo1']].rename(columns={'team1':'team','elo1':'elo'}),
                      schedule_df[['date','team2','elo2']].rename(columns={'team2':'team','elo2':'elo'})]
                      ).sort_values('date').groupby('team')['elo'].first().to_dict()

  def format_output(self, schedule_df, elo1, elo2, pred):

    return pd.DataFrame({'season'            : schedule_df['season'],
                         'game_id'           : schedule_df['game_id'],
                         'game_type'         : schedule_df['game_type'],
                         f'{self.name}_elo1' : elo1,
                         f'{self.name}_elo2' : elo2,
                         f'{self.name}_pred1': pred,
                         'result1'           : schedule_df['result1']
            })

  def prepare_data(self):
    """
    This function should generate data in the format expected by the simulate function.
    It should be run once before running simulations (not once per simulation)
    """
    raise Exception('Must overwrite prepare_data function!')

  def simulate(self):
    raise Exception('Must overwrite simulate function!')
  

# SCALE_FACTOR = 400

# class Team:
#   """Represents an entity that has an Elo-like numerical rating."""
#   def __init__(self, name: str, rating: float, K: int):
#     self.name = name
#     self.rating = rating
#     self.K = K
#     self.wins = 0
#     self.losses = 0

#   def expected_score(self, other: Self) -> float:
#     """Compute the expected score when facing the other rated entity."""
#     return 1 / (1 + 10**((other.rating - self.rating) / SCALE_FACTOR))

#   def score_delta(self, expected: float, actual: float) -> float:
#     """Compute how much the rating would change according to the given scores."""
#     # Note: Actual is 1 for win, 0 for loss, 0.5 for tie
#     return self.K * (actual - expected)
  
#   def update_score(self, expected: float, actual: float):
#     """Update the rating according to the given scores."""
#     self.rating += self.score_delta(expected, actual)
#     self.wins += actual
#     self.losses += abs(1 - actual)
#     self.K = 20 + 2**(10-self.wins-self.losses) # Update K to be more certain as season progresses

  