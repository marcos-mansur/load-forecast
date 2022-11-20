import json
import os
from typing import Dict

import pandas as pd
import tensorflow as tf
import yaml

from src.common.load_data import load_featurized_data
from src.common.logger import get_logger
from src.common.utils_evaluate import (
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
)


def predict_load(model, pred_dataset: pd.DataFrame, params: Dict):
    """If the param MODEL_TYPE is set to "AUTOREGRESSIVE" in params.yaml,
        return autoregressive predictions.

    Args:
        model (_type_): _description_
        pred_dataset (_type_): _description_
        params (_type_): _description_

    Returns:
        _type_: _description_
    """
    autoreg_steps = params["featurize"]["TARGET_PERIOD"]
    model_type = params["featurize"]["MODEL_TYPE"]
    temp_pred_dataset = pred_dataset.copy()

    if model_type == "AUTOREGRESSIVE":
        for autoreg_step in range(1, autoreg_steps + 1):
            next_prediction = model.predict(
                temp_pred_dataset.iloc[:, -autoreg_steps:], verbose=0
            )
            temp_pred_dataset[f"previsão semana {autoreg_step}"] = next_prediction

        temp_pred_dataset.index = pred_dataset.index
        print('Forecasting type: AUTOREGRESSIVE... Done!') 

    elif model_type == "SINGLE-SHOT":
        temp_pred = model.predict(temp_pred_dataset, verbose=0)
        temp_pred_dataset = temp_pred_dataset.merge(
            pd.DataFrame(
                temp_pred,
                index=pred_dataset.index,
                columns=[
                    f"previsão semana {week}" for week in range(1, autoreg_steps + 1)
                ],
            ),
            on="din_instante",
            how="outer",
        )
        print('Forecasting type: SINGLE-SHOT... Done!') 


    return temp_pred_dataset


def main():
    with open(HISTORY_PATH, "r") as history_file:
        history = json.load(history_file)
    logger = get_logger(__name__)
    load_dataset_list = load_featurized_data()

    params = yaml.safe_load(open("params.yaml"))
    window_size = params["featurize"]["WINDOW_SIZE"]
    model = tf.keras.models.load_model(
        JOB_ROOT_FOLDER / "src" / "model" / "model_train.h5"
    )

    # make prediction
    train_pred = predict_load(model, load_dataset_list["train_pred"][0], params=params)
    train_pred_date = train_pred.index + pd.Timedelta(days=7 * window_size)
    train_pred["Data Previsão"] = [str(date).split(" ")[0] for date in train_pred_date]

    val_pred = predict_load(model, load_dataset_list["val"][0], params=params)
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
