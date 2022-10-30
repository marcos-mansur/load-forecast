import tensorflow as tf
from src.const import *
import pandas as pd


def load_featurized_data():
    """
    load featurized load data, week start data and target data.
    """
    # Load energy data
    train_pred_dataset = pd.read_csv(TRAIN_PRED_PROCESSED_DATA_PATH)
    train_dataset = pd.read_csv(TRAIN_PROCESSED_DATA_PATH)
    val_dataset = pd.read_csv(VAL_PROCESSED_DATA_PATH)
    test_dataset = pd.read_csv(TEST_PROCESSED_DATA_PATH)
    load_dataset_list = {
        "train_pred": train_pred_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "train": train_dataset,
    }
    return load_dataset_list