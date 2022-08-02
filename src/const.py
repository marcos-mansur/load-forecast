# global consts
SEED = 42
REGIAO = 'SUDESTE'

# split data (preprocess.py) consts
DATA_YEAR_START_PP = 2012
DATA_YEAR_END_PP = 2022
VAL_START_PP = 0.7
TEST_START_PP = 0.9

# process.py const
BATCH_SIZE_PRO = 32
WINDOW_SIZE_PRO = 7
SUFFLE_BUFFER_PRO = 20
TARGET_PERIOD_PRO = 1
HOW_WINDOW_GEN_PRO = 'autorregressivo'

# training conts
EPOCHS = 50
PATIENCE = 100
MODEL_PATH = 'src/model/model_train.h5'
NEURONS = [32]
RUN_ID = 2


# preprocess.py debug msgs const
M_PRE_FIT = '[DEBUG] PREPROCESS - FIT (2/2): DONE!'
M_PRE_FIT_FILTER = '[DEBUG] PREPROCESS - FIT: FILTER SUBSYSTEM (1/2): DONE!'
M_PRE_FILTER = '[DEBUG] PREPROCESS - FILTER SUBSYSTEM (1/6): DONE!'
M_PRE_IMPUTE = '[DEBUG] PREPROCESS - IMPUTE NAN (2/6): DONE!'
M_PRE_GOTOFRYDAY = '[DEBUG] PREPROCESS - GO TO FRIDAY (3/6): DONE!'
M_PRE_PARSE = '[DEBUG] PREPROCESS - PARSE DATES (4/6): DONE!'
M_PRE_DROPINC = '[DEBUG] PREPROCESS - DROP INCOMPLETE WEEKS (5/6): DONE!'
M_PRE_SPLIT2 = '[DEBUG] PREPROCESS - SPLIT DATA INTO 2 FOLDS (6/6): DONE!'
M_PRE_SPLIT3 = '[DEBUG] PREPROCESS - SPLIT DATA INTO 3 FOLDS (6/6): DONE!'
# process.py debug msgs const
M_PRO_TRANS = '[DEBUG] PROCESS: TRASFORMING DATASETS (1/3): DONE!'
M_PRO_SAVE_DATA = '[DEBUG] PROCESS: SAVING DATASETS (2/3): DONE!'
M_PRO_SAVE_WEEK = '[DEBUG] PROCESS: SAVING WEEK INITIAL DAYS (3/3): DONE!'
# train.py debug msgs const
M_TRAIN_LOAD_DATA = '[DEBUG] LOAD DATA (1/5): DONE!'
M_TRAIN_LOAD_WEEK_DATA = '[DEBUG] LOAD WEEK DATA (2/5): DONE!'
M_TRAIN_LOAD_TARGET_DATA = '[DEBUG] LOAD TARGET DATA (3/5): DONE!'
M_TRAIN_CREATE_MODEL = '[DEBUG] CREATE MODEL (4/5): DONE!'
M_TRAIN_TRAINING = '[DEBUG] MODEL TRAINING (5/5): DONE!'


# preprocess.py output data path const
TREATED_DATA_PATH = 'data/treated'
TRAIN_TREATED_DATA_PATH = 'data/treated/train_preprocessed.csv'
VAL_TREATED_DATA_PATH = 'data/treated/val_preprocessed.csv'
TEST_TREATED_DATA_PATH = 'data/treated/test_preprocessed.csv'
# preprocess.py output data path const - target_df
TARGET_DF_PATH = 'data/target_df.csv'
# process.py output data path const
PROCESSED_DATA_PATH = 'data/processed'
TRAIN_PROCESSED_DATA_PATH = 'data/processed/train_processed'
TRAIN_PRED_PROCESSED_DATA_PATH = 'data/processed/train_pred_processed'
VAL_PROCESSED_DATA_PATH = 'data/processed/val_processed'
TEST_PROCESSED_DATA_PATH = 'data/processed/test_processed'
# process.py output week start day path const
TRAIN_PROCESSED_DATA_WEEK_PATH = 'data/processed/train_data_week.csv'
TRAIN_PRED_PROCESSED_DATA_WEEK_PATH = 'data/processed/train_pred_data_week.csv'
VAL_PROCESSED_DATA_WEEK_PATH = 'data/processed/val_data_week.csv'
TEST_PROCESSED_DATA_WEEK_PATH = 'data/processed/test_data_week.csv'

PROCESSED_DATA_WEEK_PATHS_LIST = [TRAIN_PRED_PROCESSED_DATA_WEEK_PATH,
                                 VAL_PROCESSED_DATA_WEEK_PATH,
                                 TEST_PROCESSED_DATA_WEEK_PATH]
