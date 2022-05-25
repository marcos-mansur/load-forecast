import pandas as pd
import optuna
import tensorflow as tf
from Functions.processing import Window_Generator
from Functions.preprocessing import Preprocessor 
from Functions.functions import (load_data,                        
                        compile_and_fit,
                        learning_curves,
                        plot_pred,
                        plot_res,
                        metrics_semana,
                        predict_load,
                        unbatch_pred,
                        create_target_df) 


time_col = 'din_instante'
load_col = 'val_cargaenergiamwmed'
SEED = 42


def create_model(trial):
    # We optimize the numbers of layers, their units and weight decay parameter.
    #weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    num_hidden = trial.suggest_int("n_units", 4, 64, log=True)
    model = tf.keras.models.Sequential([
                        # fix dimensions
                        tf.keras.layers.Lambda(lambda x: 
                                               tf.expand_dims(x, axis = -1), 
                                               input_shape=[None]),
                        # normalize data
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LSTM(num_hidden,
                                activation="tanh", 
                                #kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                return_sequences=False),
                        tf.keras.layers.Dense(1),
                        tf.keras.layers.Lambda(lambda x: x * 10000.0)
                        ])
    return model

def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = "Adam"
    kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-3, 1e-2, log=True)
    optimizer = getattr(tf.optimizers, optimizer_options)(**kwargs)
    return optimizer

def comp_fit_tuning(model, data, val_data, optimizer, epochs, patience):
    # early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    
    # compile
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=[tf.metrics.MeanAbsoluteError(),
                            tf.metrics.MeanAbsolutePercentageError(),
                            tf.keras.metrics.RootMeanSquaredError()])
    # fit data
    history = model.fit(data, epochs=epochs, verbose=0,
                        validation_data= val_data,
                        callbacks=[early_stopping]
    ) 
    return history

# load and preprocess data
df_20XX = load_data(start=2009, end=2022)

# preprocess data
pp = Preprocessor(regiao='SUDESTE')
df = pp.fit_transform(df_20XX)
# split data
train_df, val_df, test_df = pp.split_time(df=df,
                                            folds = 3,
                                            val_start=0.7, 
                                            test_start=0.9)

def generate_data(trial):

    batch_size_trial = trial.suggest_int('Batch Size', 2, 128)

    wd = Window_Generator(batch_size = batch_size_trial, 
                     window_size = trial.suggest_int('Window Size', 5, 20)*7,
                     shuffle_buffer = 20, 
                     target_period = 7, 
                     how = 'autorregressivo',
                     SEED = SEED)
    
    # dataset to training
    train_dataset, train_data_week = wd.transform(df = train_df, shuffle=True)
    val_dataset, val_data_week = wd.transform(df = val_df, shuffle=False)
    return train_dataset, val_dataset
    

def objective(trial):

    # Build model and optimizer.
    model = create_model(trial)
    optimizer = create_optimizer(trial)
    train_dataset, val_dataset = generate_data(trial)

    # Training and validating cycle.
    history = comp_fit_tuning(model=model, 
                              epochs = 150, 
                              data=train_dataset, 
                              val_data = val_dataset,
                              patience = 60, 
                              optimizer=optimizer)


    # Return loss
    return model.evaluate(val_dataset)[0]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


wd = Window_Generator(batch_size = trial.params['Batch Size'], 
                     window_size = trial.params['Window Size']*7,
                     shuffle_buffer = 20, 
                     target_period = 7, 
                     how = 'autorregressivo',
                     SEED = SEED)

# dataset to training
train_dataset, train_data_week = wd.transform(df = train_df, shuffle=True)

# dataset for performance evaluation
train_pred_dataset, train_pred_data_week = wd.transform(df = train_df, shuffle=False)
val_dataset, val_data_week = wd.transform(df = val_df, shuffle=False)
test_dataset, test_data_week = wd.transform(df = test_df, shuffle=False)
date_list = [train_pred_data_week,val_data_week,test_data_week]

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1), input_shape=[None]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(trial.params['n_units'], return_sequences=False, activation ='tanh'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 10000.0)
    ])

history = compile_and_fit(model, epochs = 150, 
                          data=train_dataset, 
                          val_data = val_dataset,
                          optimizer = tf.optimizers.Adam(learning_rate=trial.params['adam_learning_rate']),
                          patience = 100,
                          filepath = 'Models/tuned.h5')

train_pred = predict_load(model,train_pred_dataset,window_size=5)
val_pred = predict_load(model,val_dataset,window_size=5)
test_pred = predict_load(model,test_dataset,window_size=5)

pred_list = [unbatch_pred(train_pred), unbatch_pred(val_pred),unbatch_pred(test_pred)]


df_target = create_target_df(df, baseline_size=trial.params['Window Size']*7)

learning_curves(history=history, skip=20, id='best_tuned', save=True)


plot_pred(pred_list=pred_list, date_list=date_list, 
            df_target=df_target, baseline=False, 
            save=True,  id='best_tuned',)

metrics_semana(df_target, pred_list,date_list, save=True,  id='best_tuned',)

plot_res(df_target,pred_list, date_list, save=True,  id='best_tuned',)