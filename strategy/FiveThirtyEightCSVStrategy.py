import pandas as pd
import numpy as np
import util
from strategy.Strategy import Strategy

class FiveThirtyEightCSVStrategy(Strategy):
  """
  Uses the values in data/data_538.csv
  """
  def __init__(self):
    super().__init__('csv538')
  
  def prepare_data(self, seasons):
    df = util.load_data(seasons)
    return df

  def simulate(self, schedule_df):
    return self.format_output(schedule_df, schedule_df['elo1'], schedule_df['elo2'], schedule_df['elo_prob1'])

