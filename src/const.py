SEED = 42
REGIAO = 'SUDESTE'

DATA_YEAR_START_PP = 2012
DATA_YEAR_END_PP = 2022
VAL_START_PP = 0.7
TEST_START_PP = 0.9

M_PRE_FIT = '[DEBUG] PREPROCESS - FIT (1/1): DONE!'
M_PRE_FILTER = '[DEBUG] PREPROCESS - FILTER SUBSYSTEM (1/6): DONE!'
M_PRE_IMPUTE = '[DEBUG] PREPROCESS - IMPUTE NAN (2/6): DONE!'
M_PRE_GOTOFRYDAY = '[DEBUG] PREPROCESS - GO TO FRIDAY (3/6): DONE!'
M_PRE_PARSE = '[DEBUG] PREPROCESS - PARSE DATES (4/6): DONE!'
M_PRE_DROPINC = '[DEBUG] PREPROCESS - DROP INCOMPLETE WEEKS (5/6): DONE!'
M_PRE_SPLIT2 = '[DEBUG] PREPROCESS - SPLIT DATA INTO 2 FOLDS (6/6): DONE!'
M_PRE_SPLIT3 = '[DEBUG] PREPROCESS - SPLIT DATA INTO 3 FOLDS (6/6): DONE!'


BATCH_SIZE_PRO = 32
WINDOW_SIZE_PRO = 7
SUFFLE_BUFFER_PRO = 20
TARGET_PERIOD_PRO = 1
HOW_WINDOW_GEN_PRO = 'autorregressivo'


TREATED_DATA_PATH = 'data\treated'
PROCESSED_DATA_PATH = 'data\processed'

TRAIN_TREATED_DATA_PATH = 'data/treated/train_preprocessed.csv'
VAL_TREATED_DATA_PATH = 'data/treated/val_preprocessed.csv'
TEST_TREATED_DATA_PATH = 'data/treated/test_preprocessed.csv'