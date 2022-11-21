import json
import os
from typing import Dict

import mlflow
import pandas as pd
import tensorflow as tf
import yaml
from dvclive.keras import DVCLiveCallback
import matplotlib.pyplot as plt 
from datetime import datetime


from src.utils.utils_evaluate import (
    generate_metrics_semana,
    learning_curves,
    plot_predicted_series,
    plot_residual_error,
)
from src.utils.create_model import create_model
from src.utils.load_data import load_featurized_data
from src.common.logger import get_logger
from src.config.const import (
    HISTORY_PATH,
    HISTORY_PARAMS_PATH,
    TARGET_DF_PATH,
    VALUATION_PATH,
    TRAIN_MODEL_PATH,
    TRAIN_PREDICTION_PATH,
    VAL_PREDICTION_PATH,
    EVAL_ARCHIVE_PATH
)
# from src.vault_dagshub import DAGSHUB_PASSWORD, DAGSHUB_USERNAME
from src.utils.data_transform import prepare_data_for_prediction, predict_load,prepare_predicted_data
# from src.utils.vault_dagshub import get_dagshub_credentials


# mlflow settings
# DAGSHUB_USERNAME, DAGSHUB_PASSWORD = get_dagshub_credentials()
os.environ["MLFLOW_TRACKING_USERNAME"] = "ticomansur@gmail.com"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "7ab5d30298571d8f88569dedca7904ea234a18d1"
os.environ[
    "MLFLOW_TRACKING_URI"
] = "https://dagshub.com/marcos-mansur/load-forecast.mlflow"


params = yaml.safe_load(open("params.yaml"))
mlflow.tensorflow.autolog(registered_model_name=f"{params['featurize']['MODEL_TYPE']}")


plt.style.reload_library()
plt.style.use(['science','no-latex','grid',])


def compile_and_fit(
    train_data,
    val_data,
    model,
    epochs: int,
    optimizer: tf.keras.optimizers,
    filepath: str,
    batch_size: int,
    model_type: str,
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
    logger.info("Compiling model... Done!")
    
    logger.info("Training model... ")
    # fit data
    if model_type == "SINGLE-SHOT":
        history = model.fit(
            x=train_data[0],
            y=train_data[1],
            epochs=epochs,
            verbose=0,
            validation_data=val_data,
            callbacks=[early_stopping, DVCLiveCallback()],
            batch_size=batch_size,
        )

    elif model_type == "AUTOREGRESSIVE":
        history = model.fit(
            train_data,
            epochs=epochs,
            verbose=0,
            validation_data=val_data,
            callbacks=[early_stopping, DVCLiveCallback(),checkpoint],
            batch_size=batch_size,
        )
    logger.info("Training model... Done! ")

    return history

def eval(pred_list,history):
    
    logger.info("PREDICTIONS: DONE!")

    run_id = str(datetime.now()) + "_" + params['featurize']['MODEL_TYPE'] + '_' + str(params['train']['LAYERS'])

    df_target = pd.read_csv(TARGET_DF_PATH, index_col="Data")
    logger.info("LOADED TARGET DATA")
    os.makedirs(VALUATION_PATH, exist_ok=True)

    lc_fig = learning_curves(history=history.history, skip=20, plot=True)
    lc_fig.savefig(VALUATION_PATH / f"learning_curves.png")
    lc_fig.savefig(EVAL_ARCHIVE_PATH / f"learning_curves {run_id}.png")
    logger.info("LEARNING CURVES SAVED TO DISK")

    lc_fig = learning_curves(history=history.history, skip=int(len(history.history)*0.2), plot=True)
    lc_fig.savefig(VALUATION_PATH / f"learning_curves-zoom.png")
    lc_fig.savefig(EVAL_ARCHIVE_PATH / f"learning_curves-zoom {run_id} .png")
    logger.info("LEARNING CURVES SAVED TO DISK")

    # generates the plot of the original and
    # predicted time series for the 5 weeks
    pred_series_fig = plot_predicted_series(
        pred_list=pred_list, df_target=df_target, plot=True
    )
    pred_series_fig.savefig(os.path.join(VALUATION_PATH, f"prediction_series.png"))
    pred_series_fig.savefig(os.path.join(EVAL_ARCHIVE_PATH, f"prediction_series {run_id}.png"))
    logger.info("PREDICTED SERIES LINEPLOT SAVED TO DISK")

    # generates metrics for the 5 weeks and plots
    metricas_semana, metricas_fig = generate_metrics_semana(
        df_target, pred_list, plot=True
    )
    metricas_fig.savefig(os.path.join(VALUATION_PATH, f"metrics_semana.png"))
    metricas_fig.savefig(os.path.join(EVAL_ARCHIVE_PATH, f"metrics_semana {run_id}.png"))
    logger.info("WEEKLY METRICS GRAPHS SAVED TO DISK")

    train_semana_metrics = metricas_semana[0].to_dict(orient="dict")
    val_semana_metrics = metricas_semana[1].to_dict(orient="dict")
    test_semana_metrics = metricas_semana[2].to_dict(orient="dict")

    # generates the residual plot
    residual_fig, res_list = plot_residual_error(df_target, pred_list, plot=True)
    residual_fig.savefig(os.path.join(VALUATION_PATH, f"residuo.png"))
    residual_fig.savefig(os.path.join(EVAL_ARCHIVE_PATH, f"residuo {run_id}.png"))
    logger.info("RESIDUAL SAVED TO DISK")


    with open(EVAL_ARCHIVE_PATH / f"history_{run_id}", "w") as archive_path:
            json.dump(history.history, archive_path)
    with open(HISTORY_PARAMS_PATH , "w") as archive_path:
        json.dump(history.params, archive_path )
    logger.info("TRAIN HISTORY STORED TO DISK")


def main(params):
    """Main function of train module. Trains a model and saves the artifact to disk."""

    with mlflow.start_run():
        logger.info("MLFLOW RUN: STARTED!")

        model_type=params["featurize"]["MODEL_TYPE"]
        target_period=params["featurize"]["TARGET_PERIOD"]
        window_size=params["featurize"]["WINDOW_SIZE"]

        # create model
        model = create_model(params=params)
        if model:
            logger.info("CREATE MODEL: DONE!")

        load_dataset_list = load_featurized_data()

        train_data, val_data = prepare_data_for_prediction(
            load_dataset_list=load_dataset_list,
            model_type=model_type
        )

        logger.info("MODEL TRAINING: STARTING!")
        history = compile_and_fit(
            train_data=train_data,
            val_data=val_data,
            model=model,
            epochs=params["train"]["EPOCHS"],
            optimizer=tf.optimizers.Adam(),
            patience=params["train"]["PATIENCE"],
            filepath=params["train"]["MODEL_NAME"],
            batch_size=params["featurize"]["BATCH_SIZE_PRO"],
            model_type=model_type,
        )
        logger.info("MODEL TRAINING: DONE!")

        with open(HISTORY_PATH, "w") as history_file:
            json.dump(history.history, history_file)
        with open(HISTORY_PARAMS_PATH, "w") as history_params_file:
            json.dump(history.params, history_params_file)
        logger.info("TRAIN HISTORY STORED TO DISK")

        # save model to disk
        os.makedirs(TRAIN_MODEL_PATH, exist_ok=True)
        
        format = '/model_train.h5' if model_type == 'SINGLE-SHOT' else '/1/'
        
        model.save(
            TRAIN_MODEL_PATH / params["featurize"]["MODEL_TYPE"] / (params["train"]["MODEL_NAME"] + format)
        )
        logger.info("TRAINED MODEL STORED TO DISK")


        prediction_train = predict_load(model,train_data,params)
        prediction_val = predict_load(model,val_data,params)
        
        treated_pred_train = prepare_predicted_data(
            prediction_train,
            load_dataset_list['train'],
            model_type,
            target_period,
            window_size)
        treated_pred_val = prepare_predicted_data(
            prediction_val,
            load_dataset_list['val'],
            model_type,
            target_period,
            window_size)
            
        mlflow.end_run()
    logger.info("MLFLOW RUN ENDED. END OF TRAINING.")

    #################################3
    pred_list = [
        treated_pred_train,
        treated_pred_val,
    ]
    eval(pred_list=pred_list,history=history)

    treated_pred_train.to_csv(TRAIN_PREDICTION_PATH)
    treated_pred_val.to_csv(VAL_PREDICTION_PATH)


if __name__ == "__main__":
    logger = get_logger(__name__)
    main(params)
