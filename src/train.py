import os
import mlflow
from vault_dagshub import *
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from common.logger import get_logger
from const import *
from utils import *
import json

# mlflow settings
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_PASSWORD
os.environ[
    "MLFLOW_TRACKING_URI"
] = "https://dagshub.com/marcos-mansur/load-forecast.mlflow"
mlflow.tensorflow.autolog(registered_model_name=f"{REG_NAME_MODEL}")


def compile_and_fit(model, data, val_data, epochs, optimizer, filepath, patience=4):
    # early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )
    # checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
    )

    # compile
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=optimizer,
        metrics=[
            tf.metrics.MeanAbsoluteError(),
            tf.metrics.MeanAbsolutePercentageError(),
            tf.keras.metrics.RootMeanSquaredError(),
        ],
    )
    # fit data
    history = model.fit(
        data,
        epochs=epochs,
        verbose=0,
        validation_data=val_data,
        callbacks=[early_stopping],  # , checkpoint
    )
    return history


def create_model(neurons: list):

    assert (
        type(neurons) == list
    ), 'Input "neurons" to the function create_model() must be list'
    # LSTM
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Lambda(
                lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]
            ),
            tf.keras.layers.BatchNormalization(),
        ]
    )

    model.add(
        tf.keras.layers.LSTM(neurons[0], return_sequences=False, activation="tanh")
    )
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Lambda(lambda x: x * 10000.0))

    return model


def predict_load(model, pred_dataset, window_size):
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

    window_pred = []
    # loop the batches of data
    for batch_window, batch_target in pred_dataset:
        # make a copy of the window to edit
        window = batch_window
        # loop the 5 weeks we want to predict
        for week in range(0, 5):
            # predict using the most recent (window_size value) inputs
            forecast = model.predict(window[:, -window_size:], verbose=0)
            # append the prediction to the input window
            window = tf.concat(values=[window, forecast], axis=-1)
            # remove the oldest input,
            # the window_size "frame" moves one week forward (to the right)
            window = window[:, -window_size:]
        # saves only the predictions (last five columns)
        window_pred.append(window[:, -5:])
    return window_pred


def unbatch_pred(window_pred):
    """Unbatches the multi-step predictions"""

    numpy_pred = [x.numpy() for x in window_pred]
    pred = numpy_pred[0]
    for batch in numpy_pred[1:]:
        for item in batch:
            pred = np.append(pred, item).reshape([-1, 5])
    return pred[:-4]


if __name__ == "__main__":
    logger = get_logger(__name__)
    # load params
    params = yaml.safe_load(open("params.yaml"))

    load_dataset_list = load_featurized_data() 
    date_dataset_list = load_featurized_week_data()

    with mlflow.start_run():
        logger.info("MLFLOW RUN: STARTED!")

        # create model
        model = create_model(neurons=params["train"]["NEURONS"])
        if model:
            logger.info("CREATE MODEL: DONE!")

        mlflow.log_params(
            {
                "model": "vanila",
                "layers": "[LSTM]",
                "Patience": params["train"]["PATIENCE"],
                "neurons": params["train"]["NEURONS"],
                "epochs": params["train"]["EPOCHS"],
                "batch size": params["featurize"]["BATCH_SIZE_PRO"],
                "window size": params["featurize"]["WINDOW_SIZE_PRO"],
                "Process": params["featurize"]["HOW_WINDOW_GEN_PRO"],
                "Start year": params["preprocess"]["DATA_YEAR_START_PP"],
            }
        )
        logger.info("LOGGING PARAMS: DONE!")

        logger.info("MODEL TRAINING: STARTING!")
        history = compile_and_fit(
            model=model,
            epochs=params["train"]["EPOCHS"],
            data=load_dataset_list['train'],
            val_data=load_dataset_list['val'],
            optimizer=tf.optimizers.Adam(),
            patience=params["train"]["PATIENCE"],
            filepath=params["train"]["MODEL_NAME"],
        )
        logger.info("MODEL TRAINING: DONE!")
        
        with open(HISTORY_PATH,'w') as history_file:
            json.dump(history.history,history_file)
        with open(HISTORY_PARAMS_PATH,'w') as history_params_file:
            json.dump(history.params,history_params_file)
        logger.info("TRAIN HISTORY STORED TO DISK")

        # save model to disk
        os.makedirs(TRAIN_MODEL_PATH, exist_ok=True)
        model.save(os.path.join(TRAIN_MODEL_PATH, params["train"]["MODEL_NAME"]))
        logger.info("TRAINED MODEL STORED TO DISK")

        # make prediction
        train_pred = predict_load(
            model, load_dataset_list['train_pred'], window_size=params["featurize"]["WINDOW_SIZE_PRO"]
        )
        val_pred = predict_load(
            model, load_dataset_list['val'], window_size=params["featurize"]["WINDOW_SIZE_PRO"]
        )
        test_pred = predict_load(
            model, load_dataset_list['test'], window_size=params["featurize"]["WINDOW_SIZE_PRO"]
        )
        logger.info("PREDICTIONS: DONE!")

        if params['featurize']['HOW_WINDOW_GEN_PRO'] == 'autorregressivo':
            # unbatch
            pred_list = [
                unbatch_pred(train_pred),
                unbatch_pred(val_pred),
                unbatch_pred(test_pred),
            ]
            logger.info("HOW_WINDOW_GEN_PRO = autoregressivo, predictions unbatched.")
        else:
            pred_list = [
                train_pred,
                val_pred,
                test_pred,
            ]
        df_columns = ['Semana 1','Semana 2','Semana 3','Semana 4','Semana 5']
        pd.DataFrame(pred_list[0], columns=df_columns).to_csv(TRAIN_PREDICTION_DATA_PATH)
        pd.DataFrame(pred_list[1], columns=df_columns).to_csv(VAL_PREDICTION_DATA_WEEK_PATH)
        pd.DataFrame(pred_list[2], columns=df_columns).to_csv(TEST_PREDICTION_DATA_WEEK_PATH)
        logger.info('PREDICTIONS SAVED TO DISK')

        mlflow.end_run()

    logger.info("MLFLOW RUN ENDED. END OF TRAINING.")
