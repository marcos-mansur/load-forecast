from math import trunc
import os
from this import d

import pandas as pd
import tensorflow as tf
import yaml
from sklearn.base import BaseEstimator
from statsmodels.tsa.tsatools import lagmat

from src.common.logger import get_logger 
from src.const import *


class Window_Generator(BaseEstimator):
    def __init__(
        self,
        target_period,
        window_size,
        batch_size,
        shuffle_buffer,
        regiao="SUDESTE",
        sazo_weeks=2,
        SEED=SEED,
        how_input="weekly",
        how_target="weekly"
    ):
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        
        self.regiao = regiao
        self.sazo_weeks = sazo_weeks

        self.how_input = how_input
        self.how_target = how_target
        
        self.target_period = target_period 
        self.window_size = window_size
        self.target_period_mod = target_period 
        self.window_size_mod = window_size
        
        self.SEED = SEED

        # creates modfied params
        if self.how_input == 'weekly':
            self.window_size_mod = self.window_size * 7
        if self.how_input == 'daily':
            assert self.window_size%7==0, "how_input = 'daily' demmands window_size multiple of 7 (days in a week)"
            
        if self.how_target == 'weekly':
            self.target_period_mod = self.target_period * 7

        assert self.how_input in ["daily","weekly"], f"how_input only accepts 'daily' or 'weekly"
        assert self.how_target in ["daily","weekly"], f"how_target only accepts 'daily' or 'weekly"

        pass

    def generate_data_week(self, df):
        """Generate a list with the same index as the window features with the first
            day of the first week of the target
        Returns:
            data_week: first day of the first target week
        """
        df = df.copy()
        # groupby object by week and then by day
        df_grouped = df[self.window_size * 7 :].groupby(by=["semana"])["din_instante"]
        # get first day of each week and  removes the last 4 rows
        # because we want to save the first day of the first target week
        # so we drop the last target 4 weeks
        return df_grouped.min().iloc[:-4]
    

    def map_data(self, dataset):
        """Defines how the data will be processed into features and target.
        If self.how = 'input diário', the features will be in daily load
                        of window_size lenght.
        If self.how = 'sazonalidade anual', the features will be the last
                        'sazo_weeks' weeks and the same week from last year.
        if self.how = 'autorregressivo', the features are detailed in weekly
                       average load. target window_size = 1. This option
                       enables multi-step prediction.

        Args:
            dataset (tf.data.Dataset): windowed dataset

        Returns:
            tf.data.Dataset: dataset processed into features and targets
        """
        if self.how_input == "daily" and self.how_target == 'weekly':
            """inputs in daily average load, targets in weekly average load"""


            dataset = dataset.map(
                lambda window: (
                    window[: -self.target_period],  # features
                    [tf.math.reduce_mean(window[-x*7:(x-1)*7], axis=0)
                        for x in range(self.target_period)],
                    [
                        tf.math.reduce_sum(window[-35:-28]) / 7,  # first target week
                        tf.math.reduce_sum(window[-28:-21]) / 7,  # second target week
                        tf.math.reduce_sum(window[-21:-14]) / 7,  # third target week
                        tf.math.reduce_sum(window[-14:-7]) / 7,  # fourth target week
                        tf.math.reduce_sum(window[-7:]) / 7,
                    ],  # fifith target week
                )
            )
        if self.how_input == "weekly" and self.how_target == 'weekly':
            """inputs in daily average load, targets in weekly average load"""


            dataset = dataset.map(
                lambda window: (
                    
                    window[: -self.target_period],  # features
                    [tf.math.reduce_mean(window[x*7 or None:((x+1)*7)], axis=0) 
                        for x in range(0,self.window_size)
                    ],
                    tf.math.reduce_mean(window[-self.target_period :], axis=0)
                    
                )
            )
        if self.how == "sazonalidade anual":
            """the inputs are the daily load of the same week in
            the year before and in the last sazo_weeks from the time window"""
            assert (
                self.window_size * 7 >= 365
            ), """window size menor que 365 dias, 
                                        não é possível usar how = 'sazonalidade anual'"""
            assert (
                self.target_period == 35
            ), f"""targe_period = {self.target_period},
                                                deve ser igual a 35 (5 semanas)"""
            dataset = dataset.map(
                lambda window: (
                    tf.concat(
                        values=[  # week in the year before
                            tf.math.reduce_mean(
                                window[
                                    -364
                                    - self.target_period : -357
                                    - self.target_period
                                ],
                                axis=0,
                            ),
                            # last weeks
                            tf.math.reduce_mean(
                                window[
                                    -(self.sazo_weeks * 7)
                                    - self.target_period : -self.target_period
                                ],
                                axis=0,
                            ),
                        ],
                        axis=-1,
                    ),  # features
                    [
                        tf.math.reduce_sum(window[-35:-28]) / 7,  # first target week
                        tf.math.reduce_sum(window[-28:-21]) / 7,  # second target week
                        tf.math.reduce_sum(window[-21:-14]) / 7,  # third target week
                        tf.math.reduce_sum(window[-14:-7]) / 7,  # fourth target week
                        tf.math.reduce_sum(window[-7:]) / 7,
                    ],  # fifith target week
                )
            )

        if self.how == "autorregressivo":
            """for multi-step forecasting. target = next week;
            inputs = last self.window_size weeks as weekly average load
            """
            assert (
                self.target_period == 7
            ), f"""target_periodo = {self.target_period},
                                                deve ser igual = 7 (dias)"""
            assert self.window_size >= 5, "window_size menor que 5 semanas"

            dataset = dataset.map(
                lambda window: (
                    # transform input to weeks
                    tf.reshape(
                        [
                            tf.reshape(
                                tf.math.reduce_mean(
                                    window[
                                        -x * 7
                                        - self.target_period : -(x - 1) * 7
                                        - self.target_period
                                    ],
                                    axis=0,
                                ),
                                shape=[-1],
                            )
                            for x in range(int(self.window_size), 0, -1)
                        ],
                        shape=[-1],
                    ),
                    # target
                    tf.math.reduce_mean(window[-self.target_period :], axis=0),
                )
            )

        return dataset

    def group_data(self, df):
        weekly_load = df.groupby('semana')['val_cargaenergiamwmed'].mean()
        first_day_of_week = df.groupby('semana')['din_instante'].min()
        group_df = pd.DataFrame(data={'val_cargaenergiamwmed': weekly_load, 'din_instante':first_day_of_week})
        group_df_mais_dia =  pd.merge(
            left=group_df,
            right=df[['din_instante','dia semana','semana']],
            how='left',
            on='din_instante')
        return group_df_mais_dia

    def create_input_window(self, df, df_weekly):
        
        if self.how_input == 'daily':
            df_shift = df.copy()
            periodo = 'dia'
            for time_step in range(self.window_size):
                df_shift[
                    f'{periodo} {time_step+1}'
                ] = df_shift['val_cargaenergiamwmed'].shift(-time_step)
            df_shift = df_shift[df_shift['dia semana'] == 'Friday']
    
        elif self.how_input == 'weekly':
            df_shift = df_weekly.copy()
            periodo = 'semana'
            for time_step in range(self.window_size):
                df_shift[
                    f'{periodo} {time_step+1}'
                ] = df_weekly.groupby('semana')['val_cargaenergiamwmed'].mean().shift(-time_step)

        else:
            assert 1==2,"Erro aqui, doido."
        return df_shift

    
    def create_target_window(self, df, df_weekly):
        if self.how_input == 'weekly':
            time_step_factor = 7
        elif self.how_input == 'daily':
            time_step_factor = 1        
         
        if self.how_target == 'daily':
            df_shift = df.copy()
            periodo = 'dia'
             # DAILY DAILY TA QUEBRADO
            for time_step in range(self.window_size*time_step_factor,self.window_size+self.target_period):
                df_shift[
                    f'target {periodo} {time_step-self.window_size+1}'
                ] = df_shift['val_cargaenergiamwmed'].shift(-time_step)
        
            df_shift[df_shift['dia semana'] == 'Friday']


        if self.how_target == 'weekly':
            df_shift = df_weekly.copy()
            periodo = 'semana'

            if self.how_input == 'daily':
                for time_step in range(int(self.window_size/7),int(self.window_size/7)+self.target_period):
                    df_shift[
                        f'target {periodo} {time_step-(self.window_size/7)+1}'
                    ] = df_weekly.groupby('semana')['val_cargaenergiamwmed'].mean().shift(-time_step)

            if self.how_input == 'weekly':
                for time_step in range(self.window_size,self.window_size+self.target_period):
                    df_shift[
                        f'target {periodo} {time_step-(self.window_size/7)+1}'
                    ] = df_weekly.groupby('semana')['val_cargaenergiamwmed'].mean().shift(-time_step)


        return df_shift

    def transform(self, df, shuffle=True):
        """Transform a preprocessed dataframe in a windowed dataset
        Returns:
            dataset: a windowed tensorflow.dataset with window_size
                     timesteps for features and the average daily
                     load for the next five weeks as targets
        """
        df = df.copy()
        df_weekly = self.group_data(df)
        df_input_window = self.create_input_window(df,df_weekly)
        df_target_window = self.create_target_window(df,df_weekly)
        df_window_merge = pd.merge(left=df_input_window,right=df_target_window,how='left',on='din_instante',)

        # create windows
    
        if shuffle:
            # randomly shuffles the windows instances in the dataset
            df_window_merge = df_window_merge.sample(frac=1)

        return df_window_merge


def load_data_process():
    train_df = pd.read_csv(TRAIN_TREATED_DATA_PATH)
    val_df = pd.read_csv(VAL_TREATED_DATA_PATH)
    test_df = pd.read_csv(TEST_TREATED_DATA_PATH)
    return train_df, val_df, test_df


if __name__ == "__main__":

    logger = get_logger(__name__)

    train_df, val_df, test_df = load_data_process()

    params = yaml.safe_load(open("params.yaml"))["featurize"]

    wd = Window_Generator(
        batch_size=params["BATCH_SIZE_PRO"],
        window_size=params["WINDOW_SIZE_PRO"],
        shuffle_buffer=params["SUFFLE_BUFFER_PRO"],
        target_period=params["TARGET_PERIOD_PRO"],
        how_input=params["HOW_INPUT_WINDOW_GEN_PRO"],
        how_target=params["HOW_TARGET_WINDOW_GEN_PRO"],
        SEED=SEED,
    )

    # dataset for performance evaluation
    train_pred_dataset = wd.transform(df=train_df, shuffle=False)
    val_dataset = wd.transform(df=val_df, shuffle=False)
    test_dataset = wd.transform(df=test_df, shuffle=False)
    # dataset to training
    train_dataset = wd.transform(df=train_df, shuffle=True)
    logger.info("PROCESS: TRASFORMING DATASETS: DONE!")

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # saves datasets to disk
    # dataset for performance evaluation
    train_pred_dataset.to_csv(TRAIN_PRED_PROCESSED_DATA_PATH)
    val_dataset.to_csv(VAL_PROCESSED_DATA_PATH)
    test_dataset.to_csv(TEST_PROCESSED_DATA_PATH)
    # dataset to training
    train_dataset.to_csv(TRAIN_PROCESSED_DATA_PATH)
    logger.info("PROCESS: SAVING DATASETS: DONE!")
