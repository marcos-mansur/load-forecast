import pandas as pd
import yaml

from const import *
from preprocess import Preprocessor

params = yaml.safe_load(open("params.yaml"))

# load data
train_df = pd.read_csv(TRAIN_TREATED_DATA_PATH, parse_dates=["din_instante"])
val_df = pd.read_csv(VAL_TREATED_DATA_PATH, parse_dates=["din_instante"])
if params["preprocess"]["TEST_START_PP"]:
    test_df = pd.read_csv(TEST_TREATED_DATA_PATH, parse_dates="din_instante")

# print set size proportion
all_data_len = len(train_df) + len(val_df)
if params["preprocess"]["TEST_START_PP"]:
    all_data_len += len(test_df)
print("PREPROCESS DATA:")
print("TRAIN SET SIZE PROPORTION:", round(train_df.shape[0] / all_data_len, 3))
print("VAL SET SIZE PROPORTION:", round(val_df.shape[0] / all_data_len, 3))
if params["preprocess"]["TEST_START_PP"]:
    print("TEST SET SIZE PROPORTION:", round(test_df.shape[0] / all_data_len, 3))

window_size_timedelta = pd.Timedelta(
    value=params["featurize"]["WINDOW_SIZE_PRO"], unit="W"
)

print("\nTRAIN_DF:")
assert (
    train_df["dia semana"].iloc[0] == "Friday"
), "[PREPROCESS - SPLIT TIME] train_df doesn't start at a friday."
Preprocessor(params=params).check_dq(train_df)
print(
    f"First day of train_df: {train_df.din_instante.iloc[0]} - ",
    train_df["dia semana"].iloc[0],
)
print(
    f"Last day of train_df: {train_df.din_instante.iloc[-1]} - ",
    train_df["dia semana"].iloc[-1],
)

print("\nVAL_DF:")
assert (
    val_df["dia semana"].iloc[0] == "Friday"
), "[PREPROCESS - SPLIT TIME] val_df doesn't start at a friday."
assert val_df.din_instante.iloc[0] + window_size_timedelta == pd.Timestamp(
    params["preprocess"]["VAL_START_PP"]
), """
                  train_df start + window_size =! param['preprocess']['VAL_START_PP']"""
Preprocessor(params=params).check_dq(val_df)
print(
    f"First day of val_df: {val_df.din_instante.iloc[0]} - ",
    val_df["dia semana"].iloc[0],
)
print(
    f"Last day of val_df: {val_df.din_instante.iloc[-1]} - ",
    val_df["dia semana"].iloc[-1],
)

if params["preprocess"]["TEST_START_PP"]:
    print("\nTEST_DF:")
    assert (
        test_df["dia semana"].iloc[0] == "Friday"
    ), "[PREPROCESS - SPLIT TIME] test_df doesn't start at a friday."
    Preprocessor(params=params).check_dq(test_df)
    print(
        f"First day of test_df: {test_df.din_instante.iloc[0]} - ",
        test_df["dia semana"].iloc[0],
    )
    print(
        f"Last day of test_df: {test_df.din_instante.iloc[-1]} - ",
        test_df["dia semana"].iloc[-1],
    )
