""" Module for generating the target data. """

import yaml

from src.common.logger import get_logger
from src.common.target_methods import create_target_df
from src.config.const import TARGET_DF_PATH
from src.preprocess import load_and_preprocess_data

logger = get_logger(__file__)
params = yaml.safe_load(open("params.yaml"))

df, _ = load_and_preprocess_data(params=params)

create_target_df(df, df_target_path=TARGET_DF_PATH, baseline_size=5)
