import tensorflow as tf
from const import *

def load_featurized_data():
    """
    load featurized load data, week start data and target data.
    """
    # Load energy data
    train_pred_dataset = tf.data.Dataset.load(TRAIN_PRED_PROCESSED_DATA_PATH)
    train_dataset = tf.data.Dataset.load(TRAIN_PROCESSED_DATA_PATH)
    val_dataset = tf.data.Dataset.load(VAL_PROCESSED_DATA_PATH)
    test_dataset = tf.data.Dataset.load(TEST_PROCESSED_DATA_PATH)
    load_dataset_list = {
        "train_pred": train_pred_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "train": train_dataset,
    }
    return load_dataset_list