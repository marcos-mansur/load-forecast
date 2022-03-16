import os
import pandas as pd
import tensorflow as tf 
import numpy as np
import pendulum


time_col = 'din_instante'
load_col = 'val_cargaenergiamwmed'
SEED = 42

def init_constants(file_path = 'Model.h5'):
  """initialize global constants"""

  print('global constants: \n')
  
  global load_col
  load_col = 'val_cargaenergiamwmed'
  print(f'load_col: {load_col} ')
  
  global time_col
  time_col = 'din_instante'
  print(f'time_col: {time_col}')
  
  global regiao
  regiao = 'SUDESTE'
  print(f'regiao: {regiao}')

  global batch_size 
  batch_size = 32
  print(f'batch_size: {batch_size}')

  # target days to sum into weeks
  global target_period
  target_period = 7 
  print(f'target_period: {target_period}')
  
  # number of weeks in the window
  global n_weeks_ws
  n_weeks_ws = 20
  print(f'n_weeks_ws: {n_weeks_ws}')

  # window size in days for each row
  global window_size
  window_size = 7*n_weeks_ws
  print(f'window_size: {window_size}')

  global filepath
  filepath = file_path 
  print(f'filepath: {filepath}')
  
  global shuffle_buffer 
  shuffle_buffer = 20
  print(f'shuffle_buffer: {shuffle_buffer}')

  global SEED
  SEED = 42
  print(f'SEED: {SEED}')
  np.random.seed(SEED)
  tf.random.set_seed(SEED)



def load_data(start=2009, end=2021):
    """load data from ONS"""

    first_year = f'https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_{start}.csv'

    df_20XX = pd.read_csv(first_year, 
                        sep=';', 
                        parse_dates=['din_instante'])

    for x in range(start+1,end):
        df_20XX = pd.concat(objs = (df_20XX,pd.read_csv(os.path.join(f'https://ons-dl-prod-opendata.s3.amazonaws.com/',
                                                                    f'dataset/carga_energia_di/CARGA_ENERGIA_{x}.csv'), 
                            sep=';', 
                            parse_dates=['din_instante'])))
    return df_20XX



def get_missing_days(df):
  """return the missing days in the input dataset"""
  
  # range of every day from 2001 to 2021
  time_delta = pd.date_range(start = df.din_instante.iloc[0], end= df.din_instante.iloc[-1],freq='D')
  # turn into df
  df_time = pd.DataFrame(data={'data':time_delta})
  # left join range of data with datas from dataset, missing days will become NaN
  df_missing = df_time.join(df.set_index('din_instante'), on='data', how='left')
  # missing days indexes
  df_missing.val_cargaenergiamwmed[df_missing.val_cargaenergiamwmed.isnull()]
  # series of missing days with indexes
  missing_days = df_missing.loc[df_missing.val_cargaenergiamwmed[df_missing.val_cargaenergiamwmed.isnull()].index].data
  return missing_days

def id_and_impute(df):
    """input the missing day of feb'2014 with the day before and 
    drop 5 days from 2016 to complete a missing operative week"""

    # series of missing days with indexes
    missing_days = get_missing_days(df) 

    if len(df[df.din_instante == '2014-01-01']) == 1:
        # drop days from incomplete week in 2016
        df = df.drop(axis=0, index = df.din_instante[(df['din_instante']>='2016-04-01') & (df['din_instante']<='2016-04-05')].index)
        df = df.drop(axis=0, index = missing_days.index[1]-1)
        # missing day to be inputed - feb 1st, 2014
        imput_day = df['din_instante'].iloc[missing_days.index[0]]
        # line to be inserted in dataset with load value of day before
        df_day = pd.DataFrame({'din_instante': imput_day-pd.Timedelta(1, unit='D'),	
                            'val_cargaenergiamwmed': df[load_col].iloc[missing_days.index[0] - 1]},
                            index=[missing_days.index[0] -1 ])
        # insert missing day
        df = pd.concat(objs= [df[:missing_days.index[0]], 
                                    df_day, 
                                    df[missing_days.index[0]:]])
    
    #reset index
    return  df, missing_days

def go_to_friday(df): 
  """ get next friday = start the operative week"""
  
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

def treat_data(df,regiao='SUDESTE',operative_week_start=2):
  
  # round the values of load
  df['val_cargaenergiamwmed'] = np.round(df['val_cargaenergiamwmed'],2)
  # drop na rows that doesn't have load values
  df.dropna(axis=0, how='any',inplace=True)
  # filter data by subsystem region
  try:
    df = df[df.nom_subsistema==regiao].reset_index().drop('index',
                                                             axis=1).copy()
  except:
    pass
  # drops columns about region
  df.drop(labels=['nom_subsistema','id_subsistema'], 
          inplace=True, axis=1,errors='ignore')


  # check if the dataset starts on a friday and go to friday if it does not 
  df = go_to_friday(df)
  # insert missing data from 1st feb'2014
  df, _ = id_and_impute(df) 

  # create column with week number 
  df.reset_index(inplace=True,drop=True)
  df['semana'] = (df.index)//7 

  df['Mes'] = df['din_instante'].dt.month
  df['dia semana'] = df['din_instante'].dt.day_name()
  df['dia mes'] = df['din_instante'].dt.day
  df['ano'] = df['din_instante'].dt.year
  
  return df


def split_time(split_val, 
               split_test,
               df,
               regiao):
  df = df.copy()
  df = treat_data(df, regiao = regiao)
  # split datasets
  train_df = df[0:split_val].copy()
  val_df = df[split_val:split_test].copy()
  test_df = df[split_test:].copy()
  
  return (treat_data(train_df,regiao = regiao), 
         treat_data(val_df,regiao = regiao), 
         treat_data(test_df,regiao = regiao))

def time_delta(d):
  return pd.to_timedelta(d,unit='d') 


def windowed_dataset(df, batch_size, 
                     window_size, shuffle_buffer, 
                     target_period, shuffle=True):
  df = df.copy()
  
  # check if the dataset starts on a friday and go to friday if it does not 
  df = go_to_friday(df)
  # groupby object by week and then by day
  df_grouped = df[window_size:].groupby(by=['semana'])['din_instante']
  # get first day of each week
  data_week = df_grouped.min()
  # if last week is incomplete, drop it
  if df_grouped.count().iloc[-1]!=7:
    data_week = data_week[:-1]


  series = df[load_col  ]
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