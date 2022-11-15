import json
import os

import yaml

from src.common.logger import get_logger
from src.const import *
from src.utils import *


def predict_load(model, pred_dataset, params):

    autoreg_steps = params["evaluate"]["AUTOREGRESSIVE_STEPS"]
    model_type = params["evaluate"]["MODEL_TYPE"]
    temp_pred_dataset = pred_dataset.copy()

    if model_type == "AUTOREGRESSIVE":
        for autoreg_step in range(1, autoreg_steps + 1):
            next_prediction = model.predict(
                temp_pred_dataset.iloc[:, -autoreg_steps:], verbose=0
            )
            temp_pred_dataset[f"previsão semana {autoreg_step}"] = next_prediction
    temp_pred_dataset.index = pred_dataset.index

    return temp_pred_dataset


# valuation
if __name__ == "__main__":

    logger = get_logger(__name__)

    with open(HISTORY_PATH, "r") as history_file:
        history = json.load(history_file)

    load_dataset_list = load_featurized_data()

    params = yaml.safe_load(open("params.yaml"))

    model = tf.keras.models.load_model(
        JOB_ROOT_FOLDER / "src" / "model" / "model_train.h5"
    )

    # make prediction
    train_pred = predict_load(
        model,
        load_dataset_list["train_pred"][0],
        params=params,
    )
    val_pred = predict_load(
        model,
        load_dataset_list["val"][0],
        params=params,
    )
    test_pred = predict_load(
        model,
        load_dataset_list["test"][0],
        params=params,
    )
    logger.info("PREDICTIONS: DONE!")

    pred_list = [train_pred, val_pred, test_pred]
    logger.info("HOW_WINDOW_GEN_PRO != autoregressivo, no need to unbatch predictions.")

    df_target = pd.read_csv(TARGET_DF_PATH)
    logger.info("LOADED TARGET DATA")
    os.makedirs(VALUATION_PATH, exist_ok=True)

    lc_fig = learning_curves(history=history, skip=20)
    lc_fig.savefig(VALUATION_PATH / "learning_curves.png")
    logger.info("LEARNING CURVES SAVED TO DISK")

    # generates the plot of the original and
    # predicted time series for the 5 weeks
    pred_series_fig = plot_predicted_series(
        pred_list=pred_list,
        df_target=df_target,
    )
    pred_series_fig.savefig(os.path.join(VALUATION_PATH, "prediction_series.png"))
    logger.info("PREDICTED SERIES LINEPLOT SAVED TO DISK")

    # generates metrics for the 5 weeks and plots
    metricas_semana, metricas_fig = generate_metrics_semana(
        df_target,
        pred_list,
    )
    metricas_fig.savefig(os.path.join(VALUATION_PATH, "metrics_semana.png"))
    logger.info("WEEKLY METRICS GRAPHS SAVED TO DISK")

    train_semana_metrics = metricas_semana[0].to_dict(orient="dict")
    val_semana_metrics = metricas_semana[1].to_dict(orient="dict")
    test_semana_metrics = metricas_semana[2].to_dict(orient="dict")

    # generates the residual plot
    residual_fig, res_list = plot_residual_error(
        df_target,
        pred_list,
    )
    residual_fig.savefig(os.path.join(VALUATION_PATH, "residuo.png"))
    logger.info("RESIDUAL SAVED TO DISK")