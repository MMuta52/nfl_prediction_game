import nfl_data_py as nfl
import pandas as pd
import numpy as np
import time
from geopy.distance import geodesic

def clean_team_names(df):
  team_map = {'LA' :'LAR',
              'STL':'LAR',
              'SD' :'LAC',
              'OAK':'LV',
              'WSH':'WAS'}
  return df.replace(team_map)

def load_data(seasons):
  """
  Load nflfastR schedule data and merge with 538 Elo records
  """
  schedule_df = nfl.import_schedules(seasons)

  elo = pd.read_csv('data_538.csv')
  # Rename teams
  schedule_df[['home_team','away_team']] = clean_team_names(schedule_df[['home_team','away_team']])
  elo[['team1','team2']] = clean_team_names(elo[['team1','team2']])
  
  # Rename some columns
  schedule_df = schedule_df.rename(columns={'gameday':'date', 'home_team':'team1', 'away_team':'team2'})
  schedule_df.columns = [col.replace('home','team1').replace('away','team2') for col in schedule_df.columns]

  keep_cols = ['game_id', 'season', 'game_type', 'week', 'date',
  'team1', 'team2', 'div_game', 'overtime',
  'team1_moneyline', 'team2_moneyline', 'spread_line',
  'team1_spread_odds', 'team2_spread_odds', 'total_line',
  'under_odds', 'over_odds', 'team1_qb_id',
  'team2_qb_id', 'team1_qb_name', 'team2_qb_name',
  'team1_coach', 'team2_coach', 'neutral', 'playoff',
  'elo1', 'elo2', 'elo_prob1', 'score1', 'score2', 'result1']

  # Join on "teamA_teamB" string which is team1 and team2 in alphabetical order
  # Can't use home_away string because neutral field games can give mismatched ordering
  elo['teams'] = elo[['team1','team2']].apply(lambda x: '_'.join(sorted(x.values)), axis=1)
  schedule_df['teams'] = schedule_df[['team1','team2']].apply(lambda x: '_'.join(sorted(x.values)), axis=1)

  # Join schedule_df with 538 and clean up columns
  schedule_df = schedule_df.drop(labels=['team1','team2'], axis=1).merge(elo, on=['season','date','teams'])[keep_cols]
  schedule_df['date'] = pd.to_datetime(schedule_df['date'])

  return schedule_df

def get_pass_df(years):
  """
  Get weekly player data
  Filter to columns related to passing and rows where players attempted >5 passes
  """
  qbcols_538 = ['player_id', 'recent_team', 'player_name', 'season', 'week', 'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'carries', 'rushing_yards', 'rushing_tds']
  pass_df = nfl.import_weekly_data(years, columns=qbcols_538)
  pass_df = pass_df[pass_df['attempts']>5] # Exclude anyone who attempted less than 5 passes
  pass_df['recent_team'] = clean_team_names(pass_df['recent_team'])
  return pass_df.sort_values(['season','week']).reset_index(drop=True)

def score_game(pred_prob,result,game_type='REG'):
  """
  Following https://nflgamedata.com/predict/rules.php
  pred_prob: User predicted probability of result (decimal value)
  result   : 1 if predicted team wins, 0 if lost
  mult   : Multiplier which is 1 for regular season, 2 for wildcard round, 3 for divisional round, 4 for conference champ, 5 for super bowl
  """
  mult_map = {'REG':1, 'WC':2, 'DIV':3, 'CON':4, 'SB':5}
  points = mult_map[game_type] * (25.0 - (100.0 * (round(pred_prob,2) - result)**2))
  return points

def score_df(df, pred_col):
  """
  Wrapper around score_game to apply to a df of predictions. Returns a pd.Series of scores for each game prediction.
  """
  return df.apply(lambda x: score_game(x[pred_col], x['result1'], x['game_type']), axis=1)

def odds_to_implied_prob(odds):
  if odds > 0:
    return 100 / (100+abs(odds))
  else:
    return abs(odds) / (100+abs(odds))

def initialize_elos(schedule_df):
  return pd.concat([schedule_df[['date','team1','elo1']].rename(columns={'team1':'team','elo1':'elo'}),
                  schedule_df[['date','team2','elo2']].rename(columns={'team2':'team','elo2':'elo'})]
                  ).sort_values('date').groupby('team')['elo'].first().to_dict()

def get_away_travel_dist(schedule_df):
  coords = pd.read_csv('team_city_coordinates.csv', index_col='team')
  return schedule_df.apply(lambda x: geodesic(eval(coords.loc[x.team1,'coordinates']),eval(coords.loc[x.team2,'coordinates'])).mi, axis=1)

def get_rest_table(schedule_df):
  """
  Generate lokup table showing number of days of rest for a given team in a week
  """
  rest = pd.concat([schedule_df[['season','week','date','team1']].rename(columns={'team1':'team'}),
                    schedule_df[['season','week','date','team2']].rename(columns={'team2':'team'})]
                    ).pivot_table(values='date',
                                  index=['season','week'],
                                  columns='team',
                                  aggfunc='first').ffill().diff().apply(lambda x: x.dt.days).replace(0,np.nan)

  # Manually reset first week of each season to be NaN
  rest.loc[(slice(None),1),:] = np.nan
  return rest

def get_opponent_map(schedule_df):
    """
    Return series mapping each (team, season, week) to opponent played
    """
    return pd.concat([schedule_df.rename(columns={'team1':'team','team2':'opponent'}),
                      schedule_df.rename(columns={'team1':'opponent','team2':'team'})]
                      ).groupby(['season','week','team'])['opponent'].nth(0)

