import pandas as pd
from preprocess import Preprocessor as pp

train_df = pd.read_csv('data/treated/train_preprocessed.csv') 
val_df = pd.read_csv('data/treated/val_preprocessed.csv')
test_df = pd.read_csv('data/treated/test_preprocessed.csv')

print('PREPROCESS DATA:')
print('TRAIN, VALIDATION AND TEST SET SIZE PROPORTION:')
all_data_len = len(train_df) + len(val_df) + len(test_df)
print(round(train_df.shape[0]/all_data_len,3),
      round(val_df.shape[0]  /all_data_len,3), 
      round(test_df.shape[0] /all_data_len,3))

print('\nTRAIN_DF:')
assert train_df['dia semana'].iloc[0] == 'Friday', "[PREPROCESS - SPLIT TIME] train_df doesn't start at a friday."
pp().check_dq(train_df)
print(f"First day of train_df: {train_df.din_instante.iloc[0]} - ",train_df['dia semana'].iloc[0])
print(f"Last day of train_df: {train_df.din_instante.iloc[-1]} - ",train_df['dia semana'].iloc[-1])

print('\nVAL_DF:')
assert val_df['dia semana'].iloc[0] == 'Friday', "[PREPROCESS - SPLIT TIME] val_df doesn't start at a friday."
pp().check_dq(val_df)
print(f"First day of val_df: {val_df.din_instante.iloc[0]} - ", val_df['dia semana'].iloc[0])
print(f"Last day of val_df: {val_df.din_instante.iloc[-1]} - ",val_df['dia semana'].iloc[-1])

print('\nTEST_DF:')
assert test_df['dia semana'].iloc[0] == 'Friday', "[PREPROCESS - SPLIT TIME] test_df doesn't start at a friday."
pp().check_dq(test_df)
print(f"First day of test_df: {test_df.din_instante.iloc[0]} - ", test_df['dia semana'].iloc[0])
print(f"Last day of test_df: {test_df.din_instante.iloc[-1]} - ",test_df['dia semana'].iloc[-1])

