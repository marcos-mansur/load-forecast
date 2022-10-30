import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.base import BaseEstimator

from common.logger import get_logger
from const import *


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
        how="input diário",
    ):
        self.target_period = target_period * 7  # in weeks
        self.window_size = window_size  # in weeks
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.regiao = regiao
        self.SEED = SEED
        self.how = how
        self.sazo_weeks = sazo_weeks
        assert self.how in ["input diário", "sazonalidade anual", "autorregressivo"]
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
        if self.how == "input diário":
            """inputs in daily average load, targets in weekly average load"""

            assert (
                self.target_period == 35
            ), f"""targe_period = {self.target_period}, 
                                                deve ser igual a 35 (5 semanas)"""
            dataset = dataset.map(
                lambda window: (
                    window[: -self.target_period],  # features
                    [
                        tf.math.reduce_sum(window[-35:-28]) / 7,  # first target week
                        tf.math.reduce_sum(window[-28:-21]) / 7,  # second target week
                        tf.math.reduce_sum(window[-21:-14]) / 7,  # third target week
                        tf.math.reduce_sum(window[-14:-7]) / 7,  # fourth target week
                        tf.math.reduce_sum(window[-7:]) / 7,
                    ],  # fifith target week
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

    def transform(self, df, shuffle=True):
        """Transform a preprocessed dataframe in a windowed dataset
        Returns:
            dataset: a windowed tensorflow.dataset with window_size
                     timesteps for features and the average daily
                     load for the next five weeks as targets
        """
        df = df.copy()
        data_week = self.generate_data_week(df)
        series = df["val_cargaenergiamwmed"]
        # generate tf.dataset
        dataset = tf.data.Dataset.from_tensor_slices(series)

        # create windows
        dataset = dataset.window(
            self.window_size * 7 + self.target_period, shift=7, drop_remainder=True
        )
        # make sure every window is the same size / clip NaN at the end
        dataset = dataset.flat_map(
            lambda window: window.batch(self.window_size * 7 + self.target_period)
        )
        if shuffle:
            # randomly shuffles the windows instances in the dataset
            dataset = dataset.shuffle(self.shuffle_buffer, seed=self.SEED)
        # separates features and target and take the average of the target days by week
        dataset = self.map_data(dataset)

        # batch and prefetch
        dataset = dataset.batch(self.batch_size).prefetch(1)

        for i, t in dataset:
            logger.debug({"input shape": i.shape, "target shape": t.shape})
            break
        return dataset, data_week


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
        how=params["HOW_WINDOW_GEN_PRO"],
        SEED=SEED,
    )

    # dataset for performance evaluation
    train_pred_dataset, train_pred_data_week = wd.transform(df=train_df, shuffle=False)
    val_dataset, val_data_week = wd.transform(df=val_df, shuffle=False)
    test_dataset, test_data_week = wd.transform(df=test_df, shuffle=False)
    # dataset to training
    train_dataset, train_data_week = wd.transform(df=train_df, shuffle=True)
    logger.info("PROCESS: TRASFORMING DATASETS (1/3): DONE!")

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # saves datasets to disk
    # dataset for performance evaluation
    tf.data.Dataset.save(
        train_pred_dataset,
        TRAIN_PRED_PROCESSED_DATA_PATH,
        compression=None,
        shard_func=None,
        checkpoint_args=None,
    )
    tf.data.Dataset.save(
        val_dataset,
        VAL_PROCESSED_DATA_PATH,
        compression=None,
        shard_func=None,
        checkpoint_args=None,
    )
    tf.data.Dataset.save(
        test_dataset,
        TEST_PROCESSED_DATA_PATH,
        compression=None,
        shard_func=None,
        checkpoint_args=None,
    )
    # dataset to training
    tf.data.Dataset.save(
        train_dataset,
        TRAIN_PROCESSED_DATA_PATH,
        compression=None,
        shard_func=None,
        checkpoint_args=None,
    )
    logger.info("PROCESS: SAVING DATASETS (2/3): DONE!")

    # saves datasets week initial day
    # dataset for performance evaluation
    train_pred_data_week.to_csv(TRAIN_PRED_PROCESSED_DATA_WEEK_PATH)
    val_data_week.to_csv(VAL_PROCESSED_DATA_WEEK_PATH)
    test_data_week.to_csv(TEST_PROCESSED_DATA_WEEK_PATH)
    # dataset to training
    train_data_week.to_csv(TRAIN_PROCESSED_DATA_WEEK_PATH)
    logger.info("PROCESS: SAVING WEEK INITIAL DAYS (3/3): DONE!")
