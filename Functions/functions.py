import os
import pandas as pd
import tensorflow as tf 
import numpy as np
import pendulum
from sklearn.base import BaseEstimator, TransformerMixin

time_col = 'din_instante'
load_col = 'val_cargaenergiamwmed'
SEED = 42


def download_data(start=2009, end=2021):
    """load data from ONS"""

    first_year = f'https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_{start}.csv'

    df_20XX = pd.read_csv(first_year, 
                        sep=';', 
                        parse_dates=['din_instante'])

    for x in range(start+1,end+1):
        df_20XX = pd.concat(objs = (df_20XX,pd.read_csv(os.path.join(f'https://ons-dl-prod-opendata.s3.amazonaws.com/',
                                                                    f'dataset/carga_energia_di/CARGA_ENERGIA_{x}.csv'), 
                            sep=';', 
                            parse_dates=['din_instante'])))
    return df_20XX.reset_index(drop=True)


def load_data(start=2009, end=2021):
    """load data from ONS"""

    cwd_dir = os.getcwd()
     
    first_year = os.path.join(cwd_dir, f'Data/CARGA_ENERGIA_{start}.csv')

    df_20XX = pd.read_csv(first_year, 
                        sep=';', 
                        parse_dates=['din_instante'])

    for x in range(start+1,end+1):
        df_20XX = pd.concat(objs = (df_20XX,
                                    pd.read_csv(os.path.join(cwd_dir, 
                                                            f'Data/CARGA_ENERGIA_{x}.csv'), 
                            sep=';', 
                            parse_dates=['din_instante'])))
    return df_20XX.reset_index(drop=True)

def create_target_df(df):
  """ returns a dataframe with target values and baseline"""
  # average daily load by operative week
  df_target = pd.DataFrame(data=df.groupby(by=['semana'])['val_cargaenergiamwmed'].mean())
  # start day of each operative week
  df_target['Data'] = df.groupby(by=['semana'])[time_col].min()
  df_target['dia semana'] = df.groupby(by=['semana'])['dia semana'].min()
  df_target['baseline'] = df_target['val_cargaenergiamwmed'].shift(1)
  return df_target


def split_time(split_val, 
               split_test,
               df,
               regiao="SUDESTE"):

  pp = Preprocessor(regiao=regiao)
  df = pp.fit_transform(df)
  # split datasets
  train_df = df[0:split_val]
  val_df = df[split_val:split_test]
  test_df = df[split_test:]
  
  return (pp.fit_transform(train_df), 
         pp.fit_transform(val_df), 
         pp.fit_transform(test_df))


def windowed_dataset(df, batch_size,
                     window_size, shuffle_buffer, 
                     target_period, shuffle=True,
                     regiao="SUDESTE"):
  df = df.copy()
  
  if df['din_instante'].iloc[0].day_name() != 'Friday':
    # get next friday - begins the operative week
    df = Preprocessor(regiao=regiao).go_to_friday(df)
  # groupby object by week and then by day
  df_grouped = df[window_size:].groupby(by=['semana'])['din_instante']
  # get first day of each week
  data_week = df_grouped.min()
  # if last week is incomplete, drop it
  if df_grouped.count().iloc[-1]!=7:
    data_week = data_week[:-1]


  series = df[load_col]
  # generate tf.dataset
  dataset = tf.data.Dataset.from_tensor_slices(series)
  # create windows 
  dataset = dataset.window(window_size + target_period, shift=7, drop_remainder=True)
  # make sure every window is the same size / clip NaN at the end
  dataset = dataset.flat_map(lambda window: window.batch(window_size + target_period))
  if shuffle:
    # randomly shuffles the windows instances in the dataset 
    dataset = dataset.shuffle(shuffle_buffer,seed=SEED)
  # separates features and target and average the target days
  dataset = dataset.map(lambda window:(window[:-target_period], 
                                       tf.math.reduce_sum(window[-target_period:])/target_period))
  # batch and prefetch
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset, data_week


class Preprocessor(BaseEstimator, TransformerMixin):

  def __init__(self, regiao):
    self.regiao = regiao
    self.missing_days = []
    pass


  def fit(self, df:pd.DataFrame):
    """ Learns the missing days """
    df = df.copy()
    # filter by subsystem
    df = self.filter_subsystem(df, regiao = self.regiao)
    # saves missing days in a variable called missing_days 
    self.missing_days = df[pd.isna(df.val_cargaenergiamwmed)].din_instante
    return self 


  def transform(self, df:pd.DataFrame):
    """ Applies transformations """
    df = df.copy()
    df = self.filter_subsystem(df, regiao = self.regiao)  # filter by subsystem
    df = self.impute_nan(df)                              # impute/drop NaN values
    df = self.go_to_friday(df)        # starts the dataset at a friday - the operative week 
    df = self.parse_dates(df)         # create columns parsing the data
    df = self.drop_incomplete_week(df)    # drop last rows so to have full weeks
    self.check_dq(df)                   # prints the NaN values for loadand missing days
    return df


  def go_to_friday(self,df): 
    """ go next friday = begining of the operative week"""
    df = df.copy()
    # first day in dataset
    date_time = df['din_instante'].iloc[0]
    # check if the dataset starts on a friday 
    if date_time.day_name() != 'Friday':
      # today
      dt = pendulum.datetime(date_time.year,date_time.month, date_time.day)
      # next friday - begins the operative week
      next_friday = dt.next(pendulum.FRIDAY).strftime('%Y-%m-%d')
      # df starts with the begin of operative week
      df = df[df['din_instante'] >= next_friday].reset_index(drop=True).copy()
    
    return df


  def filter_subsystem(self, df:pd.DataFrame, regiao:str):
    """ filter data by subsystem and reset index """
    df = df.copy()
    # try and execept so it doesn't crash if it's applied to an already treated dataset
    try:
      df = df[df['nom_subsistema']==regiao].reset_index().drop('index',axis=1).copy()
    except:
      pass
    # dropa columns about subsystem
    df.drop(labels=['nom_subsistema','id_subsistema'], inplace=True, axis=1,errors='ignore')
    # reset index of concatenated datasets
    df.reset_index(inplace=True,drop=True)
    return df


  def parse_dates(self, df):
    """ parse date into year, month, month day and week day  """
    df = df.copy()
    
    df['semana'] = (df.index)//7
    df['dia semana'] = df['din_instante'].dt.day_name()
    df['dia mes'] = df['din_instante'].dt.day
    df['Mes'] = df['din_instante'].dt.month
    df['ano'] = df['din_instante'].dt.year
    return df

  def drop_incomplete_week(self,df):
    """ drop incomplete week at the bottom of the dataset """
    for i in range(6):
      if df['dia semana'].tail(1).item() == 'Thursday':
        break
      else:
        df.drop(labels=df.tail(1).index, axis=0, inplace=True)

    return df
  

  def impute_nan(self, df):
    """ impute the 12 NaN values """
    df = df.copy()
    time_col = 'din_instante'
    load_col = 'val_cargaenergiamwmed'
    if len(self.missing_days) != 0:
      # If the NaN weren't already dealt with:
      if df[df[time_col] == self.missing_days.iloc[0]].val_cargaenergiamwmed.isna().item():
        # impute missing day '2013-12-01' with the load from the day before
        df.at[(df[df.din_instante == self.missing_days.iloc[0]].index.item()), 
              load_col] = df[load_col].iloc[self.missing_days.index[0] - 1]
        # impute missing day '2014-02-01' with the load from the day before
        df.at[(df[df.din_instante == self.missing_days.iloc[1]].index.item()), 
              load_col] = df[load_col].iloc[self.missing_days.index[1] - 1]
        # impute missing day '2015-04-09' with the load from the day before
        df.at[(df[df.din_instante == self.missing_days.iloc[2]].index.item()), 
              load_col] = df[load_col].iloc[self.missing_days.index[2] - 1]
        # drop days from incomplete week in 2016 - from '2016-04-01' to '2016-04-14'
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.drop(axis=0, index = df[(df[time_col] >= '2016-04-01') & (df[time_col] <= '2016-04-14')].index)
    
    return df
  

  def check_dq(self,df):
    # check for NaN values
    nan_data = df[pd.isna(df.val_cargaenergiamwmed)].din_instante
    if len(nan_data) != 0:
        print("NaN values: \n")
        print(nan_data)
    else:
        print('No missing NaN.')
    
    # check for missing days in the series
    missing_days = pd.date_range(start = df.din_instante.iloc[0], 
                                 end= df.din_instante.iloc[-1],
                                 freq='D').difference(df.din_instante)
    if len(missing_days) != 0:
        print("\nMissing days in the series:")
        print(missing_days)
    else:
        print("\nNo missing days in the series")
