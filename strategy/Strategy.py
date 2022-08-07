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
                         'team1'             : schedule_df['team1'],
                         'team2'             : schedule_df['team2'],
                         'team1_moneyline'   : schedule_df['team1_moneyline'],
                         'team2_moneyline'   : schedule_df['team2_moneyline'],
                         'team2'             : schedule_df['team2'],
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
  
