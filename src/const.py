import os 

# global consts
SEED = 42
# preprocess consts
REGIAO = 'SUDESTE'

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
M_TRAIN_LOAD_DATA = '[DEBUG] LOAD DATA (1/10): DONE!'
M_TRAIN_LOAD_WEEK_DATA = '[DEBUG] LOAD WEEK DATA (2/10): DONE!'
M_TRAIN_LOAD_TARGET_DATA = '[DEBUG] LOAD TARGET DATA (3/10): DONE!'
M_TRAIN_CREATE_MODEL = '[DEBUG] CREATE MODEL (4/10): DONE!'
M_TRAIN_LOG_START = '[DEBUG] MLFLOW RUN (5/10): STARTED!'
M_TRAIN_LOG_PARAMS = '[DEBUG] LOGGING PARAMS (6/10): DONE!'
M_TRAIN_TRAINING_START = '[DEBUG] MODEL TRAINING (7/10): STARTING!'
M_TRAIN_TRAINING_END = '[DEBUG] MODEL TRAINING (8/10): DONE!'
M_TRAIN_PREDICTION = '[DEBUG] PREDICTIONS (9/10): DONE!'
M_TRAIN_LOG_METRICS = '[DEBUG] LOGGING METRICS (10/10): DONE!'


# preprocess.py output data path const
TREATED_DATA_PATH = 'data/preprocessed/'
TRAIN_TREATED_DATA_PATH = os.path.join(TREATED_DATA_PATH,'train_preprocessed.csv')
VAL_TREATED_DATA_PATH = os.path.join(TREATED_DATA_PATH,'val_preprocessed.csv')
TEST_TREATED_DATA_PATH = os.path.join(TREATED_DATA_PATH,'test_preprocessed.csv')
# preprocess.py output data path const - target_df
TARGET_DF_PATH = 'data/target/target_df.csv'
# featurized.py output data path const
PROCESSED_DATA_PATH = 'data/featurized'
TRAIN_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,'train_processed')
TRAIN_PRED_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,'train_pred_processed')
VAL_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,'val_processed')
TEST_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,'test_processed')
# process.py output week start day path const
TRAIN_PROCESSED_DATA_WEEK_PATH = os.path.join(PROCESSED_DATA_PATH,'train_data_week.csv')
TRAIN_PRED_PROCESSED_DATA_WEEK_PATH = os.path.join(PROCESSED_DATA_PATH,'train_pred_data_week.csv')
VAL_PROCESSED_DATA_WEEK_PATH = os.path.join(PROCESSED_DATA_PATH,'val_data_week.csv')
TEST_PROCESSED_DATA_WEEK_PATH = os.path.join(PROCESSED_DATA_PATH,'test_data_week.csv')

PROCESSED_DATA_WEEK_PATHS_LIST = [TRAIN_PRED_PROCESSED_DATA_WEEK_PATH,
                                 VAL_PROCESSED_DATA_WEEK_PATH,
                                 TEST_PROCESSED_DATA_WEEK_PATH]

TRAIN_MODEL_PATH = 'src/model'

VALUATION_PATH = 'src/valuation'