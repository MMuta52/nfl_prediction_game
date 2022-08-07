import pandas as pd
import numpy as np
import util
from strategy.Strategy import Strategy
from strategy.FiveThirtyEightBaseStrategy import FiveThirtyEightBaseStrategy
from strategy.FiveThirtyEightAdjustedStrategy import FiveThirtyEightAdjustedStrategy

HFA = 55.0       # Home field advantage baseline 55 elo points
HFA_COVID = 33.0 # Home field advantage is lower when there are no fans!
HFA_PRIMETIME = 30.0 # Home field advantage is stronger in prime time games
DIV_REGRESSION = 50.0 # Underdogs do better in division games
K = 25.0         # The speed at which Elo ratings change
REVERSION_FACTOR = 1/3.0 # Between seasons, a team retains 2/3 of its previous season's rating
REVERSION_BASE = 1550.0

class BestStrategy(Strategy):
  """
  Current best strategy
  """
  def __init__(self, seasons):
    super().__init__('best', seasons)

  def prepare_data(self):
    df = util.load_data(self.seasons)
    df['team2_travel_dist'] = util.get_away_travel_dist(df)
    rest = util.get_rest_table(df)
    df['team1_rest'] = df.apply(lambda x: rest.loc[(x.season,x.week),x.team1], axis=1)
    df['team2_rest'] = df.apply(lambda x: rest.loc[(x.season,x.week),x.team2], axis=1)
    df['primetime'] = util.label_primetime(df)

    # Compute win dominance
    df['win_dominance'] = df['game_id'].map(util.calc_win_dominance(self.seasons)) * np.where(df['result1']==1,1,-1)

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

    mult = (game.win_dominance+1) * (2.2 / (winner_elo_diff * 0.001 + 2.2))

    # Elo shift based on K and the margin of victory multiplier
    shift = (K * mult) * (game.result1 - prediction)

    # Apply shift
    self.elo[game.team1] += shift
    self.elo[game.team2] -= shift
  
  def format_output(self, schedule_df, elo1, elo2, qb_elo1, qb_elo2, qb_val1, qb_val2, rest, pred):
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
                         f'qb_elo1'          : qb_elo1,
                         f'qb_elo2'          : qb_elo2,
                         f'qb_val1'          : qb_val1,
                         f'qb_val2'          : qb_val2,
                         'rest_adjustment'   : rest,
                         f'{self.name}_pred1': pred,
                         'result1'           : schedule_df['result1']
            })
  
  def initialize_elos(self, schedule_df):
    return {team: REVERSION_BASE for team in schedule_df['team1'].unique()}
  
  @staticmethod
  def get_qb_adjust_df(years, opponent_map):
    pass_df = util.get_pass_df(years)
    pass_df = pass_df.merge(opponent_map, left_on=['season','week','recent_team'], right_on=['season','week','team'])
    pass_df['qb_value_epa'] = pass_df['passing_epa'].fillna(0) + pass_df['rushing_epa'].fillna(0)
    pass_df['rolling_def_val'] = pass_df.groupby('opponent')['qb_value_epa'].rolling(20).mean().groupby('opponent').shift(1).droplevel(0)
    pass_df['rolling_league_def_val'] = pass_df['qb_value_epa'].rolling(100, min_periods=10).mean().shift(1)
    # Adjust QB value for game by the defenses average "QB value allowed vs league avg"
    pass_df['def_adjusted_qb_val'] = pass_df['qb_value_epa'] + (pass_df['rolling_league_def_val'] - pass_df['rolling_def_val'])
    pass_df['rolling_qb_val'] = pass_df.groupby('player_id')['def_adjusted_qb_val'].rolling(10, min_periods=4).mean().groupby('player_id').shift(1).droplevel(0)
    pass_df['rolling_off_val'] = pass_df.groupby('recent_team')['def_adjusted_qb_val'].rolling(20).mean().groupby('recent_team').shift(1).droplevel(0)

    # Offseason reversion for QB's
    yearly_avg_qb_val = pass_df.groupby(['season'])['qb_value_epa'].mean()
    reversion_condition = ((pass_df['week']==1) & (pass_df['season']!=pass_df['season'].min()))
    pass_df.loc[reversion_condition,'rolling_qb_val'] = pass_df[reversion_condition].apply(lambda x: x.rolling_qb_val - (x.rolling_qb_val-yearly_avg_qb_val.loc[x.season-1])/4, axis=1)
    # Applying offseason reversion to offense as well. Greater magnitude than QB reversion because QB's probably more stable than an offense from season to season
    pass_df.loc[reversion_condition,'rolling_off_val'] = pass_df[reversion_condition].apply(lambda x: x.rolling_off_val - (x.rolling_off_val-yearly_avg_qb_val.loc[x.season-1])/3, axis=1)

    # Elo rating adjustment
    pass_df['qb_elo_adjustment'] = (5 * (pass_df['rolling_qb_val'] - pass_df['rolling_off_val'])).fillna(0)

    return pass_df

  def simulate(self, schedule_df):
    self.elo = self.initialize_elos(schedule_df)
    my_pred = []
    my_elo1 = []
    my_elo2 = []
    qb_elo1 = []
    qb_elo2 = []
    qb_val1 = []
    qb_val2 = []
    rest = []

    # Get QB adjustment data
    qb_adj = BestStrategy.get_qb_adjust_df(schedule_df['season'].unique().tolist(), util.get_opponent_map(schedule_df))
    schedule_df = schedule_df.merge(qb_adj[['season','week','player_id','qb_elo_adjustment','rolling_qb_val']].rename(columns={'recent_team':'team1'}),
                                    how='left',
                                    left_on=['season','week','team1_qb_id'],
                                    right_on=['season','week','player_id']).drop('player_id',axis=1).rename(columns={'qb_elo_adjustment':'team1_qb_elo_adj', 'rolling_qb_val': 'team1_qb_val'}
                            ).merge(qb_adj[['season','week','player_id','qb_elo_adjustment', 'rolling_qb_val']].rename(columns={'recent_team':'team2'}),
                                    how='left',
                                    left_on=['season','week','team2_qb_id'],
                                    right_on=['season','week','player_id']).drop('player_id',axis=1).rename(columns={'qb_elo_adjustment':'team2_qb_elo_adj', 'rolling_qb_val': 'team2_qb_val'})
    schedule_df[['team1_qb_elo_adj','team2_qb_elo_adj']] = schedule_df[['team1_qb_elo_adj','team2_qb_elo_adj']].fillna(0)
    
    for game in schedule_df.itertuples():
      # Apply offseason reversion (except in first week of simulation)
      if game.week == 1 and game.season != schedule_df['season'].min():
        FiveThirtyEightBaseStrategy.offseason_reversion(self,game)

      elo_diff = self.elo[game.team1] - self.elo[game.team2]
      
      # Home field advantage:
      # 4 points for every 1000 mi traveled by the away team (or not?)
      # Extra points if it's a primetime game
      home_field = 0
      if game.neutral == 0:
        if game.season == 2020:
          home_field = HFA_COVID# + game.team2_travel_dist * 4/1000
        else:
          home_field = HFA# + game.team2_travel_dist * 4/1000
      if game.primetime:
        home_field += HFA_PRIMETIME

      elo_diff += home_field

      # Division rivalry adjustment
      # If it's a division game, give the underdog a bit of a boost
      # if game.div_game:
      #   elo_diff -= DIV_REGRESSION * np.sign(elo_diff)

      # Rest adjustment
      rest_adjustment = FiveThirtyEightAdjustedStrategy.rest_days_to_elo(game.team1_rest) - FiveThirtyEightAdjustedStrategy.rest_days_to_elo(game.team2_rest)
      elo_diff += rest_adjustment

      # Playoff adjustment
      if game.game_type != 'REG':
        elo_diff = elo_diff * 1.2
      
      # QB adjustment
      elo_diff = elo_diff + game.team1_qb_elo_adj - game.team2_qb_elo_adj

      # Elo prediction method
      prediction = 1.0 / ((10.0 ** (-elo_diff/400.0)) + 1.0)

      # Store elos and prediction
      my_elo1.append(self.elo[game.team1])
      my_elo2.append(self.elo[game.team2])
      qb_elo1.append(game.team1_qb_elo_adj)
      qb_elo2.append(game.team2_qb_elo_adj)
      qb_val1.append(game.team1_qb_val)
      qb_val2.append(game.team2_qb_val)
      rest.append(rest_adjustment)
      my_pred.append(prediction)

      # Update Elo ratings
      self.update_elo(game, elo_diff, prediction)

    return self.format_output(schedule_df, my_elo1, my_elo2, qb_elo1, qb_elo2, qb_val1, qb_val2, rest, my_pred)
