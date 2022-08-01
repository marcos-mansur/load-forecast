import pandas as pd
import numpy as np
import pendulum
from sklearn.base import BaseEstimator, TransformerMixin
from const import *
import os

class Preprocessor(BaseEstimator, TransformerMixin):

  def __init__(self, regiao=REGIAO):
    """
    Preprocessor cleans the data to a pd.DataFrame 
    with no NaN, with weeks starting on fridays and 
    ending on thursdays and no incomplete weeks.
    """
    self.regiao = regiao
    self.missing_days = []
    pass


  def fit(self, df:pd.DataFrame,):
    """ Learns the missing days """
    df = df.copy()
    # filter by subsystem
    df = self.filter_subsystem(df, regiao = self.regiao)
    # saves missing days in a variable called missing_days 
    self.missing_days = df[pd.isna(df.val_cargaenergiamwmed)].din_instante
    print(M_PRE_FIT)
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
    print(M_PRE_FILTER, " (", REGIAO, ")")
    return df


  def impute_nan(self, df):
    """ impute the 12 NaN values """
    df = df.copy()
    if len(self.missing_days) != 0:
      # If the NaN weren't already dealt with:
      if df[df['din_instante'] == self.missing_days.iloc[0]].val_cargaenergiamwmed.isna().item():
        # impute missing day '2013-12-01' with the load from the day before
        df.at[(df[df.din_instante == self.missing_days.iloc[0]].index.item()), 
              'val_cargaenergiamwmed'] = df['val_cargaenergiamwmed'].iloc[self.missing_days.index[0] - 1]
        # impute missing day '2014-02-01' with the load from the day before
        df.at[(df[df.din_instante == self.missing_days.iloc[1]].index.item()), 
              'val_cargaenergiamwmed'] = df['val_cargaenergiamwmed'].iloc[self.missing_days.index[1] - 1]
        
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
    print(M_PRE_IMPUTE)    
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
    print(M_PRE_GOTOFRYDAY)
    return df


  def parse_dates(self, df):
    """ parse date into year, month, month day and week day  """
    df = df.copy()
    
    df['semana'] = (df.index)//7
    df['dia semana'] = df['din_instante'].dt.day_name()
    df['dia mes'] = df['din_instante'].dt.day
    df['Mes'] = df['din_instante'].dt.month
    df['ano'] = df['din_instante'].dt.year
    print(M_PRE_PARSE)
    return df


  def drop_incomplete_week(self,df):
    """ drop incomplete week at the bottom of the dataset """
    for _ in range(6):
      if df['dia semana'].tail(1).item() == 'Thursday':
        break
      else:
        df.drop(labels=df.tail(1).index, axis=0, inplace=True)
    print(M_PRE_DROPINC)
    return df
  

  def check_dq(self,df):
    # check for NaN values
    nan_data = df[pd.isna(df.val_cargaenergiamwmed)].din_instante
    if len(nan_data) != 0:
        print("NaN values: ")
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
        print("No missing days in the series")

  def split_time(self, df, val_start=0.7, test_start=None):
    """ Split dataset into train, validation and teste data

    Args:
        df (pd.DataFrame): data
        val_start (float): the proportion of the dataset where starts validation data (a number between 0 and 1, usually 0.7).
        test_start (float, optional): the proportion of the dataset where starts test data (a number between 0 and 1, usually 0.9).
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
    if test_start == None:
        folds=2
    else:
        folds=3

    if folds == 3:
      train_df = df[:split_val]
      assert train_df['dia semana'].iloc[0] == 'Friday', "[PREPROCESS - SPLIT TIME] train_df doesn't start at a friday."
      val_df = df[split_val:split_test]
      assert val_df['dia semana'].iloc[0] == 'Friday', "[PREPROCESS - SPLIT TIME] val_df doesn't start at a friday."
      test_df = df[split_test:]
      assert test_df['dia semana'].iloc[0] == 'Friday', "[PREPROCESS - SPLIT TIME] test_df doesn't start at a friday."
      print(f"First day of train_df: {train_df.din_instante.iloc[0]} - ",train_df['dia semana'].iloc[0])
      print(f"First day of val_df: {val_df.din_instante.iloc[0]} - ", val_df['dia semana'].iloc[0])
      print(f"First day of test_df: {test_df.din_instante.iloc[0]} - ", test_df['dia semana'].iloc[0])
      print(M_PRE_SPLIT3)
      return train_df, val_df, test_df

    # split datasets into train and test - 2 folds
    if folds == 2:
      train_df = df[:split_val]
      assert train_df['dia semana'].iloc[0] == 'Friday', "[PREPROCESS - SPLIT TIME] train_df doesn't start at a friday."
      val_df = df[split_val:]
      assert val_df['dia semana'].iloc[0] == 'Friday', "[PREPROCESS - SPLIT TIME] val_df doesn't start at a friday."
      print(f"First day of train_df: {train_df.din_instante.iloc[0]} - ",train_df['dia semana'].iloc[0])
      print(f"First day of val_df: {val_df.din_instante.iloc[0]} - ", val_df['dia semana'].iloc[0])
      print(M_PRE_SPLIT2)
      return train_df, val_df, None


def load_data_pp(start=2009, end=2021):
    """load data from ONS"""

    cwd_dir = os.getcwd()
     
    first_year = os.path.join(cwd_dir, f'data/raw-data/CARGA_ENERGIA_{start}.csv')

    df_20XX = pd.read_csv(first_year, 
                        sep=';', 
                        parse_dates=['din_instante'])

    for x in range(start+1,end+1):
        df_20XX = pd.concat(objs = (df_20XX,
                                    pd.read_csv(os.path.join(cwd_dir, 
                                                            f'data/raw-data/CARGA_ENERGIA_{x}.csv'), 
                            sep=';', 
                            parse_dates=['din_instante'])))
    return df_20XX.reset_index(drop=True)

def main():
    df_20XX = load_data_pp(start=DATA_YEAR_START_PP, end=DATA_YEAR_END_PP)

    pp = Preprocessor(regiao=REGIAO)
    df = pp.fit_transform(df_20XX)

    train_df, val_df, test_df = pp.split_time(df=df,
                                                val_start=VAL_START_PP, 
                                                test_start=TEST_START_PP)

    train_df.to_csv('data/treated/train_preprocessed.csv') 
    val_df.to_csv('data/treated/val_preprocessed.csv')
    test_df.to_csv('data/treated/test_preprocessed.csv')


if __name__ == '__main__':
    main()