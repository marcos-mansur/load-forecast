import mlflow
import os
from vault_dagshub import *

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_PASSWORD
os.environ["MLFLOW_TRACKING_URI"]  = "https://dagshub.com/marcos-mansur/load-forecast.mlflow"

#mlflow.tensorflow.
mlflow.tensorflow.autolog()

import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
from const import * 
from utils import *


def compile_and_fit(model, data, val_data, epochs,optimizer,
                    filepath, patience=4):
  # early stopping callback
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
  # checkpoint callback
  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath, monitor = 'loss', 
                               verbose = 0, save_best_only = True, mode = 'min')
  
  # compile
  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=optimizer,
                metrics=[tf.metrics.MeanAbsoluteError(),
                         tf.metrics.MeanAbsolutePercentageError(),
                         tf.keras.metrics.RootMeanSquaredError()])
  # fit data
  history = model.fit(data, epochs=epochs, verbose=0,
                      validation_data= val_data,
                      callbacks=[early_stopping # , checkpoint
                    ])
  return history

def create_model(neurons:list):
    
    assert type(neurons) == list, 'Input "neurons" to the function create_model() must be list'
    # LSTM
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1), input_shape=[None]),
        tf.keras.layers.BatchNormalization(),])

    model.add(tf.keras.layers.LSTM(neurons[0], return_sequences=False, activation ='tanh'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Lambda(lambda x: x * 10000.0))

    return model 

def predict_load(model,pred_dataset,window_size):
    """
    This method produces multi-step (autorecursive) 
    predictions for the avg energy load of the next five weeks.
    The input data has a number of time steps (columns) 
    defined by the arg window_size. predict_load() loops 
    over the data batches, make predictions for the 
    next week, add the predictions as a column to 
    the right (most recent) of the input data. Then, it 
    uses the most recent (window_size value) columns 
    to predict next week (five times). 

    Args:
        model (keras.engine.sequential.Sequential): predictive model
        pred_dataset (tf.python.data.experimental.ops.io._LoadDataset):
                    input data
        window_size (int): the amount of time steps used for forecasting

    Returns:
        list: next five weeks predictions 
              for each window (time series) of data
    """

    window_pred= []
    # loop the batches of data
    for batch_window, batch_target in pred_dataset:
        # make a copy of the window to edit
        window = batch_window
        # loop the 5 weeks we want to predict
        for week in range(0,5):
            # predict using the most recent (window_size value) inputs
            forecast = model.predict(window[:,-window_size:],verbose=0)
            # append the prediction to the input window
            window = tf.concat(values=[window,forecast], axis=-1)
            # remove the oldest input, 
            # the window_size "frame" moves one week forward (to the right)
            window = window[:,-window_size:]
        # saves only the predictions (last five columns) 
        window_pred.append(window[:,-5:])
    return window_pred

def unbatch_pred(window_pred):
    """ Unbatches the multi-step predictions """
    
    numpy_pred = [x.numpy() for x in window_pred]  
    pred = numpy_pred[0]
    for batch in numpy_pred[1:]:
        for item in batch:
            pred = np.append(pred,item).reshape([-1,5])
    return pred[:-4]

# load params
params = yaml.safe_load(open("params.yaml"))

# load data
train_pred_dataset = tf.data.experimental.load(TRAIN_PRED_PROCESSED_DATA_PATH, 
                element_spec=None, compression=None, reader_func=None)
train_dataset = tf.data.experimental.load(TRAIN_PROCESSED_DATA_PATH, 
                element_spec=None, compression=None, reader_func=None)
val_dataset = tf.data.experimental.load(VAL_PROCESSED_DATA_PATH, 
                element_spec=None, compression=None, reader_func=None)
test_dataset = tf.data.experimental.load(TEST_PROCESSED_DATA_PATH, 
                element_spec=None, compression=None, reader_func=None)
print(M_TRAIN_LOAD_DATA)

# load week initial days data
train_pred_data_week = pd.read_csv( TRAIN_PRED_PROCESSED_DATA_WEEK_PATH, 
                                    index_col='semana')
train_data_week = pd.read_csv(TRAIN_PROCESSED_DATA_WEEK_PATH, 
                                    index_col='semana')
val_data_week = pd.read_csv(VAL_PROCESSED_DATA_WEEK_PATH, 
                                    index_col='semana')
test_data_week = pd.read_csv(TEST_PROCESSED_DATA_WEEK_PATH, 
                                    index_col='semana')
date_list = [train_pred_data_week,val_data_week,test_data_week]
print(M_TRAIN_LOAD_WEEK_DATA)

# load target data
df_target = pd.read_csv(TARGET_DF_PATH)
print(M_TRAIN_LOAD_TARGET_DATA)


with mlflow.start_run():
    print(M_TRAIN_LOG_START)

    # create model
    model = create_model(neurons= params["train"]['NEURONS'])
    if model:
        print(M_TRAIN_CREATE_MODEL)

    mlflow.log_params(
        {   
            'model': 'vanila', 
            'layers': '[LSTM]',
            'Patience': params["train"]['PATIENCE'],
            'neurons':  params["train"]['NEURONS'],
            'epochs':  params["train"]['EPOCHS'],
            'batch size': params['featurize']['BATCH_SIZE_PRO'],
            'window size': params['featurize']['WINDOW_SIZE_PRO'],
            'Process': params['featurize']['HOW_WINDOW_GEN_PRO'],
            'Start year': params['preprocess']['DATA_YEAR_START_PP']
        })
    print(M_TRAIN_LOG_PARAMS)

    print(M_TRAIN_TRAINING_START)
    history = compile_and_fit(model=model, epochs = params["train"]['EPOCHS'], 
                            data = train_dataset, 
                            val_data = val_dataset,
                            optimizer = tf.optimizers.Adam(),
                            patience = params["train"]['PATIENCE'],
                            filepath = params["train"]['MODEL_NAME'])
    print(M_TRAIN_TRAINING_END)

    # save model to disk
    os.makedirs(TRAIN_MODEL_PATH, exist_ok=True)
    model.save(os.path.join(TRAIN_MODEL_PATH, 
                            params["train"]['MODEL_NAME']))

    # make prediction
    train_pred = predict_load(model,train_pred_dataset,
                            window_size=params['featurize']['WINDOW_SIZE_PRO'])
    val_pred = predict_load(model,val_dataset,
                            window_size=params['featurize']['WINDOW_SIZE_PRO'])
    test_pred = predict_load(model,test_dataset,
                            window_size=params['featurize']['WINDOW_SIZE_PRO'])
    print(M_TRAIN_PREDICTION)

    # unbatch
    pred_list = [unbatch_pred(train_pred), 
                unbatch_pred(val_pred),
                unbatch_pred(test_pred)]


    # valuation
    learning_curves(
        history=history, 
        skip=20, 
        save=True, 
        id=''
        )

    # generates the plot of the original and
    # predicted time series for the 5 weeks
    plot_predicted_series(  
        pred_list=pred_list, 
        date_list=date_list, 
        df_target=df_target, 
        baseline=False, 
        save=True, 
        id=''
        )

    # generates metrics for the 5 weeks and plots
    metricas_semana = generate_metrics_semana(
                                df_target, 
                                pred_list,
                                date_list, 
                                save=True, 
                                id=''
                                )

    train_semana_metrics = metricas_semana[0].to_dict(orient='dict')
    val_semana_metrics = metricas_semana[1].to_dict(orient='dict')
    test_semana_metrics = metricas_semana[2].to_dict(orient='dict')
    
    metrica ='RMSE' #, 'MAE', 'MAPE']:
    mlflow.log_metrics({f'[train] {metrica}: {semana}': carga 
                    for semana,carga in train_semana_metrics[metrica].items()})
    mlflow.log_metrics({f'[val] {metrica}:{semana}': carga 
                    for semana,carga in val_semana_metrics[metrica].items()})
    mlflow.log_metrics({f'[test] {metrica}:{semana}': carga 
                    for semana,carga in test_semana_metrics[metrica].items()})
    print(M_TRAIN_LOG_METRICS)

    # generates the residual plot
    plot_residual_error(
        df_target,
        pred_list, 
        date_list, 
        save=True, 
        id=''
        )
    mlflow.end_run()

print('Bye!')