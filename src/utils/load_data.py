""" Module with methods to load data in different stages of transformation """

import os

import pandas as pd

from src.config.const import (
    TEST_PROCESSED_DATA_PATH,
    TEST_TREATED_DATA_PATH,
    TRAIN_PRED_PROCESSED_DATA_PATH,
    TRAIN_PROCESSED_DATA_PATH,
    TRAIN_TREATED_DATA_PATH,
    VAL_PROCESSED_DATA_PATH,
    VAL_TREATED_DATA_PATH,
)


def load_raw_data(start=2009, end=2021):
    """load data from ONS"""

    cwd_dir = os.getcwd()

    first_year = os.path.join(cwd_dir, f"data/raw-data/CARGA_ENERGIA_{start}.csv")

    df_20XX = pd.read_csv(first_year, sep=";", parse_dates=["din_instante"])

    for x in range(start + 1, end + 1):
        df_20XX = pd.concat(
            objs=(
                df_20XX,
                pd.read_csv(
                    os.path.join(cwd_dir, f"data/raw-data/CARGA_ENERGIA_{x}.csv"),
                    sep=";",
                    parse_dates=["din_instante"],
                ),
            )
        )
    return df_20XX.reset_index(drop=True)


def load_processed_data(params=None):
    """Loads preprocessed data.

    Returns:
        pd.DataFrame: list of preprocessed data dataframes.
    """
    train_df = pd.read_csv(
        TRAIN_TREATED_DATA_PATH, index_col="din_instante", parse_dates=True
    )
    val_df = pd.read_csv(
        VAL_TREATED_DATA_PATH, index_col="din_instante", parse_dates=True
    )

    if params["preprocess"]["TEST_START_PP"]:
        test_df = pd.read_csv(
            TEST_TREATED_DATA_PATH, index_col="din_instante", parse_dates=True
        )
    else:
        test_df = None
    return train_df, val_df, test_df


def load_featurized_data():
    """
    load featurized load data, week start data and target data.
    """
    # Load energy data
    train_pred_dataset_x = pd.read_csv(
        TRAIN_PRED_PROCESSED_DATA_PATH, index_col="din_instante", parse_dates=True
    ).filter(like="input semana")
    train_pred_dataset_y = pd.read_csv(
        TRAIN_PRED_PROCESSED_DATA_PATH, index_col="din_instante", parse_dates=True
    ).filter(like="target semana")

    train_dataset_x = pd.read_csv(
        TRAIN_PROCESSED_DATA_PATH, index_col="din_instante", parse_dates=True
    ).filter(like="input semana")
    train_dataset_y = pd.read_csv(
        TRAIN_PROCESSED_DATA_PATH, index_col="din_instante", parse_dates=True
    ).filter(like="target semana")

    val_dataset_x = pd.read_csv(
        VAL_PROCESSED_DATA_PATH, index_col="din_instante", parse_dates=True
    ).filter(like="input semana")
    val_dataset_y = pd.read_csv(
        VAL_PROCESSED_DATA_PATH, index_col="din_instante", parse_dates=True
    ).filter(like="target semana")

    test_dataset_x = pd.read_csv(
        TEST_PROCESSED_DATA_PATH, index_col="din_instante", parse_dates=True
    ).filter(like="input semana")
    test_dataset_y = pd.read_csv(
        TEST_PROCESSED_DATA_PATH, index_col="din_instante", parse_dates=True
    ).filter(like="target semana")

    load_dataset_list = {
        "train_pred": [train_pred_dataset_x, train_pred_dataset_y],
        "val": [val_dataset_x, val_dataset_y],
        "test": [test_dataset_x, test_dataset_y],
        "train": [train_dataset_x, train_dataset_y],
    }
    return load_dataset_list
