""" Module for transforming cleaned data into properly featurized input data """
import os

import pandas as pd
import yaml
from sklearn.base import BaseEstimator

from src.common.load_data import load_processed_data
from src.common.logger import get_logger
from src.config.const import (
    PROCESSED_DATA_PATH,
    TEST_PROCESSED_DATA_PATH,
    TRAIN_PRED_PROCESSED_DATA_PATH,
    TRAIN_PROCESSED_DATA_PATH,
    VAL_PROCESSED_DATA_PATH,
)


class WindowGenerator(BaseEstimator):
    """Class for generating windowed dataframes for time series prediction"""

    def __init__(
        self,
        target_period,
        window_size,
        batch_size,
        shuffle_buffer,
        regiao="SUDESTE",
        sazo_weeks=2,
        how_input="weekly",
        how_target="weekly",
        model_type='AUTOREGRESSIVE'
    ):
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

        self.regiao = regiao
        self.sazo_weeks = sazo_weeks

        self.how_input = how_input
        self.how_target = how_target

        self.window_size = window_size
        self.target_period_mod = target_period

        self.model_type = model_type

        # creates modfied params
        if model_type == 'SINGLE-STEP':
            self.target_period = target_period
        elif model_type == 'AUTOREGRESSIVE':
            self.target_period = 1

        if self.how_input == "daily":
            assert (
                self.window_size % 7 == 0
            ), "how_input = 'daily' demmands window_size multiple of 7 (days in a week)"        

        assert self.how_input in [
            "daily",
            "weekly",
        ], "how_input only accepts 'daily' or 'weekly"
        assert self.how_target in [
            "daily",
            "weekly",
        ], "how_target only accepts 'daily' or 'weekly"

        pass

    def group_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Groups load data by weekly average

        Args:
            df (pd.DataFrame): input data

        Returns:
            pd: DataFrame: input dataframe aggregated by average weekly load
        """
        weekly_load = df.groupby("semana")["val_cargaenergiamwmed"].mean()
        # reset index so we can use din_instante in the calculations
        df_reset_index = df.reset_index(drop=False)
        first_day_of_week = df_reset_index.groupby("semana")["din_instante"].min()
        group_df = pd.DataFrame(
            data={
                "val_cargaenergiamwmed": weekly_load,
                "din_instante": first_day_of_week,
            }
        )
        group_df_mais_dia = pd.merge(
            left=group_df,
            right=df_reset_index[["din_instante", "dia semana", "semana"]],
            how="left",
            on="din_instante",
        )
        return group_df_mais_dia

    def create_input_window(
        self, df: pd.DataFrame, df_weekly: pd.DataFrame
    ) -> pd.DataFrame:
        """Creates a input window dataframe from a series.

        Args:
            df (pd.DataFrame): input data series dataframe
            df_weekly (pd.DataFrame): input data series dataframe
                aggregated by average weekly load

        Returns:
            pd.DataFrame: a windowed input dataframe with shifted values.
        """

        if self.how_input == "daily":
            df_shift = df.copy()
            periodo = "dia"
            for time_step in range(self.window_size):
                df_shift[f"input {periodo} {time_step+1}"] = df_shift[
                    "val_cargaenergiamwmed"
                ].shift(-time_step)

        elif self.how_input == "weekly":
            df_shift = df_weekly.copy()
            periodo = "semana"
            for time_step in range(1, self.window_size + 1):
                df_shift[f"input {periodo} {time_step}"] = df_weekly[
                    "val_cargaenergiamwmed"
                ].shift(-time_step + 1)

        return df_shift

    def create_target_window(
        self, df: pd.DataFrame, df_weekly: pd.DataFrame
    ) -> pd.DataFrame:
        """Creates a target window dataframe from a series.

        Args:
            df (pd.DataFrame): data series dataframe
            df_weekly (pd.DataFrame): data series dataframe
                aggregated by average weekly load

        Returns:
            pd.DataFrame: a windowed dataframe with shifted values for target.
        """

        if self.how_input == "weekly":
            time_step_factor = 7
        elif self.how_input == "daily":
            time_step_factor = 1

        if self.how_target == "daily":
            df_shift = df.copy()
            periodo = "dia"
            # DAILY DAILY TA QUEBRADO
            for time_step in range(
                self.window_size * time_step_factor,
                self.window_size + self.target_period,
            ):
                df_shift[
                    f"target {periodo} {time_step/time_step_factor-self.window_size+1}"
                ] = df_shift["val_cargaenergiamwmed"].shift(-time_step)

            df_shift[df_shift["dia semana"] == "Friday"]

        if self.how_target == "weekly":
            df_shift = df_weekly.copy()
            periodo = "semana"

            if self.how_input == "daily":
                for time_step in range(
                    int(self.window_size / 7),
                    int(self.window_size / 7) + self.target_period,
                ):
                    df_shift[
                        f"target {periodo} {time_step-int(self.window_size/7)+1}"
                    ] = (
                        df_weekly.groupby("semana")["val_cargaenergiamwmed"]
                        .mean()
                        .shift(-time_step)
                    )

            if self.how_input == "weekly":
                for time_step in range(
                    self.window_size, self.window_size + self.target_period
                ):
                    df_shift[
                        f"target {periodo} {time_step-int(self.window_size/7)+1}"
                    ] = df_weekly["val_cargaenergiamwmed"].shift(-time_step - 1)

        return df_shift

    def transform(self, df: pd.DataFrame, shuffle: bool = True) -> pd.DataFrame:
        """Transform a preprocessed dataframe in a windowed dataset
        Returns:
            dataset: a windowed tensorflow.dataset with window_size
                     timesteps for features and the average daily
                     load for the next five weeks as targets
        """
        df = df.copy()
        df_period = df.copy()

        if self.how_input == "weekly" or self.how_target == "weekly":
            df_period = self.group_data(df)

        df_input_window = self.create_input_window(df, df_period)
        df_target_window = self.create_target_window(df, df_period)

        df_window_merge = pd.merge(
            left=df_input_window, right=df_target_window, how="left", on="din_instante"
        )
        df_window_merge.set_index("din_instante", inplace=True)

        if shuffle:
            # randomly shuffles the windows instances in the dataset
            df_window_merge = df_window_merge.sample(frac=1)

        df_drop_na = df_window_merge.dropna(axis=0, how="any")
        df_drop_cols = df_drop_na.drop(
            labels=[
                "Unnamed: 0",
                "val_cargaenergiamwmed_x",
                "val_cargaenergiamwmed_y",
                "dia semana_x",
                "dia semana_y",
                "semana_x",
                "semana_y",
                "ano",
                "Mes",
                "dia mes",
            ],
            errors="ignore",
            axis=1,
        )
        return df_drop_cols


def main():
    """Main function of featurize module. Featurize cleaned data and save it to disk."""

    params = yaml.safe_load(open("params.yaml"))

    train_df, val_df, test_df = load_processed_data(params)
    logger.info("FEATURIZE: LOADING DATA... DONE!")

    wd = WindowGenerator(
        batch_size=params["featurize"]["BATCH_SIZE_PRO"],
        window_size=params["featurize"]["WINDOW_SIZE"],
        shuffle_buffer=params["featurize"]["SUFFLE_BUFFER_PRO"],
        target_period=params["featurize"]["TARGET_PERIOD"],
        how_input=params["featurize"]["HOW_INPUT_WINDOW_GEN"],
        how_target=params["featurize"]["HOW_TARGET_WINDOW_GEN"],
        model_type=params["featurize"]["MODEL_TYPE"]
    )

    # dataset for performance evaluation
    train_pred_dataset = wd.transform(df=train_df, shuffle=False)
    val_dataset = wd.transform(df=val_df, shuffle=False)
    #test_dataset = wd.transform(df=test_df, shuffle=False)
    # dataset to training
    train_dataset = wd.transform(df=train_df, shuffle=True)
    logger.info("FEATURIZE: TRASFORMING DATASETS... DONE!")

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # saves datasets to disk
    # dataset for performance evaluation
    train_pred_dataset.to_csv(TRAIN_PRED_PROCESSED_DATA_PATH, index="din_instante")
    val_dataset.to_csv(VAL_PROCESSED_DATA_PATH, index="din_instante")
    #test_dataset.to_csv(TEST_PROCESSED_DATA_PATH, index="din_instante")
    # dataset to training
    train_dataset.to_csv(TRAIN_PROCESSED_DATA_PATH, index="din_instante")
    logger.info("FEATURIZE: SAVING DATASETS... DONE!")


if __name__ == "__main__":
    logger = get_logger(__name__)
    main()
