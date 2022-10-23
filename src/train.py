import json
import os

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from common.logger import get_logger
from const import *
from utils import *
from utils_tf import load_featurized_data
from vault_dagshub import *

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
            data=load_dataset_list["train"],
            val_data=load_dataset_list["val"],
            optimizer=tf.optimizers.Adam(),
            patience=params["train"]["PATIENCE"],
            filepath=params["train"]["MODEL_NAME"],
        )
        logger.info("MODEL TRAINING: DONE!")

        with open(HISTORY_PATH, "w") as history_file:
            json.dump(history.history, history_file)
        with open(HISTORY_PARAMS_PATH, "w") as history_params_file:
            json.dump(history.params, history_params_file)
        logger.info("TRAIN HISTORY STORED TO DISK")

        # save model to disk
        os.makedirs(TRAIN_MODEL_PATH, exist_ok=True)
        model.save(os.path.join(TRAIN_MODEL_PATH, params["train"]["MODEL_NAME"]))
        logger.info("TRAINED MODEL STORED TO DISK")


        mlflow.end_run()

    logger.info("MLFLOW RUN ENDED. END OF TRAINING.")
