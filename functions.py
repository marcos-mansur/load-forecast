import os
import pandas as pd
import tensorflow as tf 
import numpy as np
import pendulum
from sklearn.base import BaseEstimator, TransformerMixin

time_col = 'din_instante'
load_col = 'val_cargaenergiamwmed'
SEED = 42


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


class Preprocessor(BaseEstimator, TransformerMixin):

  def __init__(self, regiao, input_form):
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


def create_target_df(df, baseline_size=1):
  """ returns a dataframe with target values and baseline"""
  # average daily load by operative week
  df_target = pd.DataFrame(data=df.groupby(by=['semana'])['val_cargaenergiamwmed'].mean())
  # start day of each operative week
  df_target.columns = ['Semana 1']
  df_target['Semana 2'] = df_target['Semana 1'].shift(-1)
  df_target['Semana 3'] = df_target['Semana 1'].shift(-2)
  df_target['Semana 4'] = df_target['Semana 1'].shift(-3)
  df_target['Semana 5'] = df_target['Semana 1'].shift(-4)
  # defines the first day of Semana 1
  df_target['Data'] = df.groupby(by=['semana'])['din_instante'].min()
  df_target['dia semana'] = df.groupby(by=['semana'])['dia semana'].min()
  df_target['Resíduo'] = df_target['Semana 2']  - df_target['Semana 1']
  df_target['Média Móvel'] = df_target['Semana 1'].shift(1).rolling(baseline_size).mean()
  return df_target


class Baseline(tf.keras.Model):
  
  """ If how = "last week", pass the last week five times as prediction
      for the next five weeks. 
      IF how = "five weeks" past the last 5 weeks of the input window 
      as predictions for the next five 
  """
  
  def __init__(self,how):
    super().__init__()
    self.how = how
    pass
  
  def call(self, inputs):

    # last five weeks as prediction for the next five, respectivamente
    if self.how == 'five weeks':
      result =[tf.math.reduce_sum(inputs[0,-35:-28])/7, # first target week - five weeks before
              tf.math.reduce_sum(inputs[0,-28:-21])/7, # second target week - five weeks before
              tf.math.reduce_sum(inputs[0,-21:-14])/7, # third target week - five weeks before
              tf.math.reduce_sum(inputs[0,-14:-7])/7,  # fourth target week - five weeks before
              tf.math.reduce_sum(inputs[0,-7:])/7    # fifith target week - five weeks before
              ]     
    
    # last week five times as prediction for the next five weeks  
    elif self.how == 'last week':
      result=[tf.math.reduce_sum(inputs[0,-7:])/7, # first target week - five weeks before
              tf.math.reduce_sum(inputs[0,-7:])/7, # second target week - five weeks before
              tf.math.reduce_sum(inputs[0,-7:])/7, # third target week - five weeks before
              tf.math.reduce_sum(inputs[0,-7:])/7,  # fourth target week - five weeks before
              tf.math.reduce_sum(inputs[0,-7:])/7    # fifith target week - five weeks before
              ]     

    return result


class Window_Generator(BaseEstimator):
    
    def __init__(self, target_period, window_size, batch_size,shuffle_buffer,
                 regiao, SEED=SEED, how = 'dia para semana'):
        self.target_period = target_period
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.regiao = regiao
        self.SEED = SEED
        self.how = how
        assert self.how in ['dia para semana', 'sazonalidade anual']
        assert self.window_size % 7 == 0, "window_size deve ser divisível de 7"
    pass
    
    def generate_data_week(self, df):
        """Generate a list with the same index as the window features with the first
            day of the first week of the target
        Returns:
            data_week: first day of the first target week
        """
        df = df.copy()
        # if df['din_instante'].iloc[0].day_name() != 'Friday':
        #     # get next friday - begins the operative week
        #     df = Preprocessor(regiao=self.regiao).go_to_friday(df)
        # groupby object by week and then by day
        df_grouped = df[self.window_size:].groupby(by=['semana'])['din_instante']
        # get first day of each week and  removes the last 4 rows 
        # because we want to save the first day of the first target week
        # so we drop the last target 4 weeks 
        return df_grouped.min()[:-4]
    
    def map_data(self, dataset):
        """Defines how the data will be processed into features and target.
        If self.how = 'dia para semana', the features will be in daily load of window_size lenght.
        If self.how = 'sazonalidade anual', the features will be the last 4 weeks and the same week from last year.

        Args:
            dataset (tf.data.Dataset): windowed dataset

        Returns:
            tf.data.Dataset: dataset processed into features and targets
        """
        if self.how == 'dia para semana':
            dataset = dataset.map(lambda window:(window[:-self.target_period],   #features
                                        [tf.math.reduce_sum(window[-35:-28])/7, # first target week
                                        tf.math.reduce_sum(window[-28:-21])/7, # second target week
                                        tf.math.reduce_sum(window[-21:-14])/7, # third target week
                                        tf.math.reduce_sum(window[-14:-7])/7,  # fourth target week
                                        tf.math.reduce_sum(window[-7:])/7]      # fifith target week
                                                )
                                 )
        
        if self.how == 'sazonalidade anual':
            assert self.window_size >= 365, "window size menor que 365 dias, não é possível usar how = 'sazonalidade anual'"

            dataset = dataset.map(lambda window:(tf.concat(values=[window[-364-self.target_period:-357-self.target_period],
                                                                   window[-14-self.target_period:-self.target_period]],
                                                            axis=-1),   #features
                                        [tf.math.reduce_sum(window[-35:-28])/7, # first target week
                                        tf.math.reduce_sum(window[-28:-21])/7, # second target week
                                        tf.math.reduce_sum(window[-21:-14])/7, # third target week
                                        tf.math.reduce_sum(window[-14:-7])/7,  # fourth target week
                                        tf.math.reduce_sum(window[-7:])/7]      # fifith target week
                                                )
                                 )

        return dataset

    def transform(self, df, shuffle=True):
        """Transform a preprocessed dataframe in a windowed dataset
        Returns:
            dataset: a windowed tensorflow.dataset with window_size timesteps for features
                     and the average daily load for the next five weeks as targets
        """
        df = df.copy()
        data_week = self.generate_data_week(df)

        series = df['val_cargaenergiamwmed']
        # generate tf.dataset
        dataset = tf.data.Dataset.from_tensor_slices(series)
        
        # create windows 
        dataset = dataset.window(self.window_size + self.target_period, shift=7, drop_remainder=True)
        # make sure every window is the same size / clip NaN at the end
        dataset = dataset.flat_map(lambda window: window.batch(self.window_size + self.target_period))
        if shuffle:
            # randomly shuffles the windows instances in the dataset 
            dataset = dataset.shuffle(self.shuffle_buffer,seed=self.SEED)
        # separates features and target and take the average of the target days by week
        dataset = self.map_data(dataset)

        # batch and prefetch
        dataset = dataset.batch(self.batch_size).prefetch(1)
        return dataset, data_week


def compile_and_fit(model, data, val_data, epochs,optimizer,
                    filepath, patience=4):
  # early stopping callback
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
  # checkpoint callback
  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath, monitor = 'loss', 
                               verbose = 1, save_best_only = True, mode = 'min')
  
  # compile
  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=optimizer,
                metrics=[tf.metrics.MeanAbsoluteError(),
                         tf.metrics.MeanAbsolutePercentageError(),
                         tf.keras.metrics.RootMeanSquaredError()])
  # fit data
  history = model.fit(data, epochs=epochs, verbose=0,
                      validation_data= val_data,
                      callbacks=[early_stopping # , checkpoint
                    ])
  return history


def learning_curves(history, skip):
  # starting epoch to plot
  skip = 20

  fig,ax = plt.subplots(figsize=(30,6), nrows=1, ncols=3)
  metrics_list = ['loss',
                  'val_loss',
                  'mean_absolute_error', 
                  'val_mean_absolute_error',
                  'root_mean_squared_error',
                  'val_root_mean_squared_error'
                  ]

  for i, metric in enumerate(metrics_list):
    if i<=1:
      ax1 =  ax.ravel()[0]
    elif i>1 and i<=3:
      ax1 =  ax.ravel()[1]
    else:
      ax1= ax.ravel()[2]
    sns.lineplot(x = range(skip,len(history.history[metric])),
                y = history.history[metric][skip:],
                ax = ax1)


  ax.ravel()[0].set_title("Learning Curve: MSE - loss")
  ax.ravel()[0].legend(labels=['Treino', 'Validação'])
  ax.ravel()[1].set_title("Learning Curve: MAE")
  ax.ravel()[1].legend(labels=['Treino', 'Validação'])
  ax.ravel()[2].set_title("Learning Curve: RMSE")
  ax.ravel()[2].legend(labels=['Treino', 'Validação'])

  plt.show()


def plot_pred(date_list, pred_list, df_target, baseline=False):
    
    colors = ['orange', 'green', 'purple']
    _,ax=plt.subplots(figsize=(20,35), ncols=1, nrows=5)
    extra = plt.Rectangle((0, 0), 0, 0, fc="none", fill=False, ec='none', linewidth=0)

    if baseline == True:
        sns.lineplot(x = df_target['Data'],
                y = df_target['Média Móvel'], 
                ax=np.ravel(ax)[0],
                color='black')


    for x in range(0,5):
        # plot measured data
        sns.lineplot(x = df_target['Data'], 
                    y = df_target[f'Semana {x+1}'], 
                    ax=np.ravel(ax)[x], 
                    color = 'teal')

        # plot predicted data
        for date,pred,color in zip(date_list, 
                                pred_list,
                                colors): 
            sns.lineplot(x = date[:].shift(x),
                        y = pred[:,x], 
                        ax=np.ravel(ax)[x],
                        color=color)
                        

        np.ravel(ax)[x].set_title(f'Carga real vs Predição em todo o período - Semana {x+1}')
        # np.ravel(ax)[x].legend(['Real','Previsão no treino','Previsão na validação','Previsão no teste'], loc='upper left')

        
        scores = (r"MAE Train ={:.0f}"+'\n'+r"MAE val ={:.0f}"+"\n"+r"MAE test ={:.0f}").format(
                tf.keras.metrics.mean_absolute_error(df_target[f'Semana {x+1}'].loc[np.array(date_list[0].index)],
                                                    train_pred[:,x]).numpy(),
                tf.keras.metrics.mean_absolute_error(df_target[f'Semana {x+1}'].loc[np.array(date_list[1].index)],
                                                    val_pred[:,x]).numpy(),
                tf.keras.metrics.mean_absolute_error(df_target[f'Semana {x+1}'].loc[np.array(date_list[2].index)],
                                                    test_pred[:,x]).numpy() 
                                                                                                )
        np.ravel(ax)[x].legend([extra], [scores], loc='lower right')
    plt.show()


def plot_res(df_target,pred_list, date_list):
    # variation from one week to the next
    res_baseline = df_target['Resíduo'] #- df_target['Resíduo'].mean())/df_target['Resíduo'].max()
    # prediction residues
    colors = ['orange', 'green', 'purple']


    fig, ax =plt.subplots(figsize=(20,8))
    # diferença normalizada entre semanas consecutivas
    #sns.lineplot(y=res_baseline, x=df_target['Data'], ax=ax)

    for pred, date, color in zip(pred_list,date_list,colors):
        
        res_pred = pred[:,0] - df_target[f'Semana 1'].loc[np.array(date.index)]
        sns.lineplot(y=res_pred, x=df_target['Data'], ax=ax, color= color)

    ax.set_title("Resíduo - Semana 1")
    ax.legend('')
    plt.show()


def metrics_semana(df_target, pred_list,date_list):
    name_dict = {'0':'train_data',
                '1':'val_data',
                '2': 'test_data'}

    fig, ax = plt.subplots(figsize=(20,10),ncols=3,nrows=3)

    for [i,df_loop,data_week] in zip([0,1,2],pred_list, date_list):

        mae_list = []
        mape_list = []
        mse_list = []

        for x in range(0,5):
            #print(f"MAE train_pred Semana {i+1}:  {mean_absolute_error(train_pred[:,i], df_target[5 : int(len(df_target)*0.7)-4].iloc[:,i])}")
            mae_list.append(mean_absolute_error(df_loop[:,x], 
                                                df_target[f'Semana {x+1}'].loc[np.array(data_week.index)]))
        for x in range(0,5):
            #print(f"MAPE train_pred Semana {i+1}: {mean_absolute_percentage_error(train_pred[:,i], df_target[5 : int(len(df_target)*0.7)-4].iloc[:,i])*100}")
            mape_list.append(mean_absolute_percentage_error(df_loop[:,x], 
                                                            df_target[f'Semana {x+1}'].loc[np.array(data_week.index)])*100)
        for x in range(0,5):
            #print(f"MSE train_pred Semeana {i+1}: {mean_squared_error(train_pred[:,i], df_target[5 : int(len(df_target)*0.7)-4].iloc[:,i])}")                
            mse_list.append(mean_squared_error(df_loop[:,x], df_target[f'Semana {x+1}'].loc[np.array(data_week.index)]))
        

        # rectangle to print the metrics mean and std over it 
        extra = plt.Rectangle((0, 0), 0, 0, fc="none", fill=False, ec='none', linewidth=0)
        # plot MAE by week
        sns.lineplot(x=range(1,6),y=mae_list, ax=ax[i,0])
        ax[i,0].set_title(f'{name_dict[str(i)]} - MAE por semana prevista')
        ax[i,0].set_xticks([1,2,3,4,5])
        scores = (r'$MAE={:.2f} \pm {:.2f}$').format(np.mean(mae_list),np.std(mae_list))
        ax[i,0].legend([extra], [scores], loc='lower right')
        # plot mape by week
        sns.lineplot(x=range(1,6),y=mape_list, ax=ax[i,1])
        ax[i,1].set_title(f'{name_dict[str(i)]} - MAPE por semana prevista')
        ax[i,1].set_xticks([1,2,3,4,5])
        scores = (r'$MAPE={:.2f} \pm {:.2f}$').format(np.mean(mape_list),np.std(mape_list))
        ax[i,1].legend([extra], [scores], loc='lower right')
        # plot MSE by week
        sns.lineplot(x=range(1,6),y=mse_list, ax=ax[i,2])
        ax[i,2].set_title(f'{name_dict[str(i)]} - MSE por semana prevista')
        ax[i,2].set_xticks([1,2,3,4,5])
        scores = (r'$MSE={:.2f} \pm {:.2f}$').format(np.mean(mse_list),np.std(mse_list))
        ax[i,2].legend([extra], [scores], loc='lower right')

    plt.show()


def baseline_metrics(df_target, date_list):
    # metrics for baseline model
    metrics_baseline = pd.DataFrame(index = ['train', 'val', 'test'], 
                        data = 

            {'MAE' : [tf.keras.metrics.mean_absolute_error(
            # target
            df_target[f'Semana 1'].loc[np.array(date_list[x].index)],
            # baseline (week before)
            df_target[f'Média Móvel'].loc[np.array(date_list[x].index)]).numpy()
            for x in range(0,3)],
            
            'MAPE' : [tf.keras.metrics.mape(
            # target
            df_target[f'Semana 1'].loc[np.array(date_list[x].index)],
            # baseline (week before)
            df_target[f'Média Móvel'].loc[np.array(date_list[x].index)]).numpy()
            for x in range(0,3)],

            'MSE' : [tf.keras.metrics.mse(
             # target
            df_target[f'Semana 1'].loc[np.array(date_list[x].index)],
            # baseline (week before)
            df_target[f'Média Móvel'].loc[np.array(date_list[x].index)]).numpy()
            for x in range(0,3)],

            'MSLE' : [tf.keras.metrics.msle(
            # target
            df_target[f'Semana 1'].loc[np.array(date_list[x].index)],
            # baseline (week before)
            df_target[f'Média Móvel'].loc[np.array(date_list[x].index)]).numpy()
            for x in range(0,3)],

            'RMSE' : [tf.keras.metrics.RootMeanSquaredError().update_state(
                        # target
            df_target[f'Semana 1'].loc[np.array(date_list[x].index)],
            # baseline (week before)
            df_target[f'Média Móvel'].loc[np.array(date_list[x].index)]).numpy()
            for x in range(0,3)]
            })
    return metrics_baseline



