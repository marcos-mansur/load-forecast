import numpy as np
import tensorflow as tf

from Functions.preprocessing import Preprocessor
from Functions.processing import Window_Generator
from Functions.functions import (load_data,                        
                        create_target_df,
                        compile_and_fit,
                        learning_curves,
                        plot_pred,
                        plot_res,
                        metrics_semana,
                        predict_load,
                        unbatch_pred,
                        ) 


regiao = 'SUDESTE'
SEED = 42

batch_size = 32
# number of week to be predicted
target_period = 1
# window size in weeks for each row
window_size = 5
# optmizer learning rate
adam_learning_rate = 0.007500032882345478

filepath = 'Models/Model_v3.h5'
id = 2 

np.random.seed(SEED)
tf.random.set_seed(SEED)


df_20XX = load_data(start=2012, end=2022)


pp = Preprocessor(regiao='SUDESTE')
df = pp.fit_transform(df_20XX)

train_df, val_df, test_df = pp.split_time(df=df,
                                            folds = 3,
                                            val_start=0.7, 
                                            test_start=0.9)

df_target = create_target_df(df, baseline_size=5)

wd = Window_Generator(batch_size = batch_size, 
                     window_size = 7,
                     shuffle_buffer = 20, 
                     target_period = 1, 
                     how = 'autorregressivo',
                     SEED = SEED)


# dataset to training
train_dataset, train_data_week = wd.transform(df = train_df, shuffle=True)

# dataset for performance evaluation
train_pred_dataset, train_pred_data_week = wd.transform(df = train_df, shuffle=False)
val_dataset, val_data_week = wd.transform(df = val_df, shuffle=False)
test_dataset, test_data_week = wd.transform(df = test_df, shuffle=False)



date_list = [train_pred_data_week,val_data_week,test_data_week]


# LSTM
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1), input_shape=[None]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(32, return_sequences=False, activation ='tanh'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 10000.0)
    ])


history = compile_and_fit(model, epochs = 150, 
                          data=train_dataset, 
                          val_data = val_dataset,
                          optimizer = tf.optimizers.Adam(), #learning_rate=0.001
                          patience = 100,
                          filepath = filepath)

train_pred = predict_load(model,train_pred_dataset,window_size=5)
val_pred = predict_load(model,val_dataset,window_size=5)
test_pred = predict_load(model,test_dataset,window_size=5)

pred_list = [unbatch_pred(train_pred), unbatch_pred(val_pred),unbatch_pred(test_pred)]

learning_curves(history=history, skip=20, save=True, id=id)


plot_pred(pred_list=pred_list, date_list=date_list, df_target=df_target, baseline=False, save=True,id=id)

metrics_semana(df_target, pred_list,date_list, save=True, id=id)

plot_res(df_target,pred_list, date_list, save=True, id=id)
