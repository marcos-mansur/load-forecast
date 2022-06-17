import pandas as pd
import numpy as np
import pendulum
from sklearn.base import BaseEstimator, TransformerMixin

time_col = 'din_instante'
load_col = 'val_cargaenergiamwmed'
SEED = 42


class Preprocessor(BaseEstimator, TransformerMixin):

  def __init__(self, regiao, input_form=None):
    self.regiao = regiao
    self.missing_days = []
    self.input_form = input_form
    pass


  def fit(self, df:pd.DataFrame,):
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


  def filter_subsystem(self, df:pd.DataFrame, regiao:str):
    """ filter data by subsystem and reset index """
    df = df.copy()
    # try and except so it doesn't crash if it's applied to an already treated dataset
    try:
      df = df[df['nom_subsistema']==regiao].reset_index().drop('index',axis=1).copy()
    except:
      pass
    # dropa columns about subsystem
    df.drop(labels=['nom_subsistema','id_subsistema'], inplace=True, axis=1,errors='ignore')
    # reset index of concatenated datasets
    df.reset_index(inplace=True,drop=True)
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
        
        # Impute 2015-04-09
        # variation between april 9 (wednesday) and april 10 (thursday) of 2014
        var2014 = df[df['din_instante'] == r'2014-04-10'].val_cargaenergiamwmed.item() /  df[df['din_instante'] == r'2014-04-09'].val_cargaenergiamwmed.item()
        # index of 2015-04-09, the missing day in 2015
        index_2015 = df[df['din_instante'] == r'2015-04-09'].index.item()
        # replace the missing day of 2015 with the day before * variation between same days in 2014
        df.loc[index_2015, 'val_cargaenergiamwmed'] = df[df['din_instante'] == r'2015-04-08'].val_cargaenergiamwmed.item()*var2014

        # Impute missing days from 2016-04-05 to 2016-04-13
        # list of daily variations between 2015-04-07 and 2015-04-16  
        var_2015 =[]
        for dia in range(7,16):
            var_2015.append(df[df['din_instante'] == r'2015-04-{:0>2d}'.format(dia)].val_cargaenergiamwmed.item() / 
                        df[df['din_instante'] == r'2015-04-{:0>2d}'.format(dia-1)].val_cargaenergiamwmed.item())
        # index of 2016-04-05, the begining of the 9 missing days in 2016
        index_2016 = df[df['din_instante'] == r'2016-04-05'].index.item()

        for x in range(0,9):
            df.loc[index_2016 + x,'val_cargaenergiamwmed'] = df[df['din_instante'] == r'2015-04-{:0>2d}'.format(x+7)].val_cargaenergiamwmed.item()*var_2015[x]
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

  def split_time(self, df, val_start=0.7, test_start=0.9, folds=3):
    """ Split dataset into train, validation and teste data

    Args:
        df (pd.DataFrame): data
        val_start (float, optional): the proportion of the dataset where starts validation data. Defaults to 0.7.
        test_start (float, optional): the proportion of the dataset where starts test data. Defaults to 0.9.
        regiao (str, optional): Subsystem to filter data. Defaults to "SUDESTE".

    Returns:
        pd.DataFrame: train data, validation data and test data dataframes 
    """
    df = df.copy()
    # index of end of training dataset, start of validation dataset
    split_val = int(len(df)*val_start)
    # make sure we split the dataset on a friday - first day of the operative week
    for x in range(0,7):
        # Check if the last day is friday, then the day before, then before...
        if df.loc[split_val-x,'dia semana'] == 'Friday':
            # when we find the friday before the split, we update the split index
            split_val = split_val - x
            break
    # index of end of validation dataset, start of test dataset
    split_test = int(len(df)*test_start)
    # make sure we split the dataset on a friday - first day of the operative week
    for i in range(0,7):
      # Check if the last day is friday, then the day before, then before...
        if df.loc[split_test-i,'dia semana'] == 'Friday':
            # when we find the friday before the split, we update the split index
            split_test = split_test - i
            break    
    # split datasets into train, validation and test  3 folds
    if folds == 3:
      train_df = df[:split_val]
      val_df = df[split_val:split_test]
      test_df = df[split_test:]
      print(f"First day of train_df: {train_df.din_instante.iloc[0]}")
      print(f"First day of val_df: {val_df.din_instante.iloc[0]}")
      print(f"First day of test_df: {test_df.din_instante.iloc[0]}")
      return train_df, val_df, test_df

    # split datasets into train and test - 2 folds
    if folds == 2:
      train_df = df[:split_val]
      val_df = df[split_val:]
      print(f"First day of train_df: {train_df.din_instante.iloc[0]}")
      print(f"First day of val_df: {val_df.din_instante.iloc[0]}")
      return train_df, val_df
