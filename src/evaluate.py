import json
import os
from typing import Dict

import pandas as pd
import tensorflow as tf
import yaml

from src.utils.data_transform import prepare_data_for_prediction, predict_load
from src.utils.load_data import load_featurized_data
from src.common.logger import get_logger
from src.utils.utils_evaluate import (
    generate_metrics_semana,
    learning_curves,
    plot_predicted_series,
    plot_residual_error,
)
from src.config.const import (
    HISTORY_PATH,
    JOB_ROOT_FOLDER,
    TARGET_DF_PATH,
    VALUATION_PATH,
    TRAIN_MODEL_PATH
)



def main():

    with open(HISTORY_PATH, "r") as history_file:
        history = json.load(history_file)

    logger = get_logger(__name__)

    params = yaml.safe_load(open("params.yaml"))
    window_size = params["featurize"]["WINDOW_SIZE"]
    model_name = params["train"]["MODEL_NAME"]
    model_type = params["featurize"]["MODEL_TYPE"]

    
    load_dataset_list = load_featurized_data()
    logger.info('Loading data... Done!')
    
    model_path = JOB_ROOT_FOLDER / "src" / "model" / model_name
    format = 'model_train.h5' if model_type == 'SINGLE-SHOT' else '/1/'

    if model_type == 'SINGLE-SHOT':
        model = tf.keras.models.load_model(
            model_path
        )
    elif model_type == 'AUTOREGRESSIVE':
        model = tf.saved_model.load(
            TRAIN_MODEL_PATH / params["featurize"]["MODEL_TYPE"] / (params["train"]["MODEL_NAME"] + format)
        )

    load_dataset_list = prepare_data_for_prediction(load_dataset_list,model_type)
    # make prediction
    train_pred = predict_load(model, load_dataset_list[0], params=params)
    train_pred_date = train_pred.index + pd.Timedelta(days=7 * window_size)
    train_pred["Data Previsão"] = [str(date).split(" ")[0] for date in train_pred_date]

    val_pred = predict_load(model, load_dataset_list[1], params=params)
    val_pred_date = val_pred.index + pd.Timedelta(days=7 * window_size)
    val_pred["Data Previsão"] = [str(date).split(" ")[0] for date in val_pred_date]
    
    #    test_pred = predict_load(model, load_dataset_list["test"][0], params=params)
    
    pred_list = [
        train_pred,
        val_pred,
        #    test_pred
    ]
    logger.info("PREDICTIONS: DONE!")

    df_target = pd.read_csv(TARGET_DF_PATH, index_col="Data")
    logger.info("LOADED TARGET DATA")
    os.makedirs(VALUATION_PATH, exist_ok=True)

    lc_fig = learning_curves(history=history, skip=20, plot=True)
    lc_fig.savefig(VALUATION_PATH / "learning_curves.png")
    logger.info("LEARNING CURVES SAVED TO DISK")

    # generates the plot of the original and
    # predicted time series for the 5 weeks
    pred_series_fig = plot_predicted_series(
        pred_list=pred_list, df_target=df_target, plot=True
    )
    pred_series_fig.savefig(os.path.join(VALUATION_PATH, "prediction_series.png"))
    logger.info("PREDICTED SERIES LINEPLOT SAVED TO DISK")

    # generates metrics for the 5 weeks and plots
    metricas_semana, metricas_fig = generate_metrics_semana(
        df_target, pred_list, plot=True
    )
    metricas_fig.savefig(os.path.join(VALUATION_PATH, "metrics_semana.png"))
    logger.info("WEEKLY METRICS GRAPHS SAVED TO DISK")

    train_semana_metrics = metricas_semana[0].to_dict(orient="dict")
    val_semana_metrics = metricas_semana[1].to_dict(orient="dict")
    test_semana_metrics = metricas_semana[2].to_dict(orient="dict")

    # generates the residual plot
    residual_fig, res_list = plot_residual_error(df_target, pred_list, plot=True)
    residual_fig.savefig(os.path.join(VALUATION_PATH, "residuo.png"))
    logger.info("RESIDUAL SAVED TO DISK")


# valuation
if __name__ == "__main__":

    main()
