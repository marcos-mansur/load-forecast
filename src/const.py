""" Module with constants """

from pathlib import Path

# registered model name
REG_NAME_MODEL = "temp"

# global consts
SEED = 42
# preprocess consts
REGIAO = "SUDESTE"

JOB_ROOT_FOLDER: Path = (
    Path(__file__ if __file__ is not None else ".").resolve().parent.parent
)

# preprocess.py output data path const
TREATED_DATA_PATH: Path = JOB_ROOT_FOLDER / "data" / "preprocessed"
TRAIN_TREATED_DATA_PATH: Path = TREATED_DATA_PATH / "train_preprocessed.csv"
VAL_TREATED_DATA_PATH: Path = TREATED_DATA_PATH / "val_preprocessed.csv"
TEST_TREATED_DATA_PATH: Path = TREATED_DATA_PATH / "test_preprocessed.csv"
# preprocess.py output data path const - target_df
TARGET_DF_PATH: Path = JOB_ROOT_FOLDER / "data" / "target" / "target_df.csv"
# featurized.py output data path const
PROCESSED_DATA_PATH: Path = JOB_ROOT_FOLDER / "data" / "featurized"
TRAIN_PROCESSED_DATA_PATH: Path = PROCESSED_DATA_PATH / "train_processed.csv"
TRAIN_PRED_PROCESSED_DATA_PATH: Path = PROCESSED_DATA_PATH / "train_pred_processed.csv"

VAL_PROCESSED_DATA_PATH: Path = PROCESSED_DATA_PATH / "val_processed.csv"
TEST_PROCESSED_DATA_PATH: Path = PROCESSED_DATA_PATH / "test_processed.csv"

PREDICTION_DATA_PATH: Path = JOB_ROOT_FOLDER / "data" / "predicted"

TRAIN_PREDICTION_DATA_PATH: Path = PREDICTION_DATA_PATH / "train.csv"
VAL_PREDICTION_DATA_WEEK_PATH: Path = PREDICTION_DATA_PATH / "val.csv"
TEST_PREDICTION_DATA_WEEK_PATH: Path = PREDICTION_DATA_PATH / "test.csv"

TRAIN_MODEL_PATH = JOB_ROOT_FOLDER / "src" / "model"

VALUATION_PATH: Path = JOB_ROOT_FOLDER / "evaluation"
HISTORY_PATH: Path = VALUATION_PATH / "history.json"
HISTORY_PARAMS_PATH: Path = VALUATION_PATH / "history_params.json"
