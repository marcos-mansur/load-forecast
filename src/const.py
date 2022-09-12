import os

import yaml

# registered model name
REG_NAME_MODEL = yaml.safe_load(open("params.yaml"))["featurize"]["HOW_WINDOW_GEN_PRO"]

# global consts
SEED = 42
# preprocess consts
REGIAO = "SUDESTE"

# preprocess.py output data path const
TREATED_DATA_PATH = "data/preprocessed/"
TRAIN_TREATED_DATA_PATH = os.path.join(TREATED_DATA_PATH, "train_preprocessed.csv")
VAL_TREATED_DATA_PATH = os.path.join(TREATED_DATA_PATH, "val_preprocessed.csv")
TEST_TREATED_DATA_PATH = os.path.join(TREATED_DATA_PATH, "test_preprocessed.csv")
# preprocess.py output data path const - target_df
TARGET_DF_PATH = "data/target/target_df.csv"
# featurized.py output data path const
PROCESSED_DATA_PATH = "data/featurized"
TRAIN_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "train_processed")
TRAIN_PRED_PROCESSED_DATA_PATH = os.path.join(
    PROCESSED_DATA_PATH, "train_pred_processed"
)
VAL_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "val_processed")
TEST_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "test_processed")
# process.py output week start day path const
TRAIN_PROCESSED_DATA_WEEK_PATH = os.path.join(
    PROCESSED_DATA_PATH, "train_data_week.csv"
)
TRAIN_PRED_PROCESSED_DATA_WEEK_PATH = os.path.join(
    PROCESSED_DATA_PATH, "train_pred_data_week.csv"
)
VAL_PROCESSED_DATA_WEEK_PATH = os.path.join(PROCESSED_DATA_PATH, "val_data_week.csv")
TEST_PROCESSED_DATA_WEEK_PATH = os.path.join(PROCESSED_DATA_PATH, "test_data_week.csv")

PROCESSED_DATA_WEEK_PATHS_LIST = [
    TRAIN_PRED_PROCESSED_DATA_WEEK_PATH,
    VAL_PROCESSED_DATA_WEEK_PATH,
    TEST_PROCESSED_DATA_WEEK_PATH,
]

PREDICTION_DATA_PATH = "data/predicted"

TRAIN_PREDICTION_DATA_PATH = os.path.join(PREDICTION_DATA_PATH, "train.csv")
VAL_PREDICTION_DATA_WEEK_PATH = os.path.join(PREDICTION_DATA_PATH, "val.csv")
TEST_PREDICTION_DATA_WEEK_PATH = os.path.join(PREDICTION_DATA_PATH, "test.csv")

TRAIN_MODEL_PATH = "src/model"

VALUATION_PATH = "evaluation/"
HISTORY_PATH = "evaluation/history.json"
HISTORY_PARAMS_PATH = "evaluation/history_params.json"
