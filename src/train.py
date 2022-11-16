import json
import os
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.common.load_data import load_featurized_data
from src.common.logger import get_logger
from src.config.const import (
    HISTORY_PARAMS_PATH,
    HISTORY_PATH,
    REG_NAME_MODEL,
    TRAIN_MODEL_PATH,
)
from src.vault_dagshub import DAGSHUB_USERNAME, DAGSHUB_PASSWORD

# mlflow settings
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_PASSWORD
os.environ[
    "MLFLOW_TRACKING_URI"
] = "https://dagshub.com/marcos-mansur/load-forecast.mlflow"
mlflow.tensorflow.autolog(registered_model_name=f"{REG_NAME_MODEL}")


def compile_and_fit(
    x: pd.DataFrame,
    y: pd.DataFrame,
    val_data: pd.DataFrame,
    model,
    epochs: int,
    optimizer: tf.keras.optimizers,
    filepath: str,
    batch_size: int,
    patience: int = 4,
) -> Dict:
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
        x=x,
        y=y,
        epochs=epochs,
        verbose=0,
        validation_data=(val_data[0], val_data[1]),
        callbacks=[early_stopping],  # , checkpoint
        batch_size=batch_size,
    )
    return history


def create_model(
    neurons: list, 
    model_type: str,
    target_period: int,
) -> tf.keras.models.Sequential:

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

    if model_type == 'AUTOREGRESSIVE':
        last_layer_neurons = 1
    if model_type == 'SINGLE-STEP':
        last_layer_neurons = target_period
    
    model.add(tf.keras.layers.Dense(last_layer_neurons))
    model.add(tf.keras.layers.Lambda(lambda x: x * 10000.0))

    return model


def main():
    """Main function of train module. Trains a model and saves the artifact to disk."""

    params = yaml.safe_load(open("params.yaml"))
    load_dataset_list = load_featurized_data()

    with mlflow.start_run():
        logger.info("MLFLOW RUN: STARTED!")

        # create model
        model = create_model(
            neurons=params["train"]["NEURONS"], 
            model_type = params["featurize"]["MODEL_TYPE"],
            target_period = params["featurize"]["TARGET_PERIOD"]
        )
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
                "window size": params["featurize"]["WINDOW_SIZE"],
                "window_input_pp": params["featurize"]["HOW_INPUT_WINDOW_GEN"],
                "window_target_pp": params["featurize"]["HOW_TARGET_WINDOW_GEN"],
                "Start year": params["preprocess"]["DATA_YEAR_START_PP"],
            }
        )
        logger.info("LOGGING PARAMS: DONE!")

        logger.info("MODEL TRAINING: STARTING!")
        history = compile_and_fit(
            x=load_dataset_list["train"][0],
            y=load_dataset_list["train"][1],
            model=model,
            epochs=params["train"]["EPOCHS"],
            val_data=load_dataset_list["val"],
            optimizer=tf.optimizers.Adam(),
            patience=params["train"]["PATIENCE"],
            filepath=params["train"]["MODEL_NAME"],
            batch_size=params["featurize"]["BATCH_SIZE_PRO"],
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


if __name__ == "__main__":
    logger = get_logger(__name__)
    main()
