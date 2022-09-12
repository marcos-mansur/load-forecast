import json
import os

import tensorflow

from common.logger import get_logger
from const import *
from utils import *

# valuation
if __name__ == "__main__":

    logger = get_logger(__name__)

    with open(HISTORY_PATH, "r") as history_file:
        history = json.load(history_file)

    pred_list = load_prediction_data()
    logger.info("LOAD ENERGY DATA: DONE!")
    date_dataset_list = load_featurized_week_data()
    logger.info("LOAD WEEK DATA: DONE!")

    df_target = pd.read_csv(TARGET_DF_PATH)
    os.makedirs(VALUATION_PATH, exist_ok=True)

    lc_fig = learning_curves(history=history, skip=20)
    lc_fig.savefig(os.path.join(VALUATION_PATH, "learning_curves.png"))

    # generates the plot of the original and
    # predicted time series for the 5 weeks
    pred_series_fig = plot_predicted_series(
        pred_list=pred_list,
        date_list=date_dataset_list,
        df_target=df_target,
        baseline=False,
    )
    pred_series_fig.savefig(os.path.join(VALUATION_PATH, "prediction_series.png"))

    # generates metrics for the 5 weeks and plots
    metricas_semana, metricas_fig = generate_metrics_semana(
        df_target,
        pred_list,
        date_dataset_list,
    )
    metricas_fig.savefig(os.path.join(VALUATION_PATH, "metrics_semana.png"))

    train_semana_metrics = metricas_semana[0].to_dict(orient="dict")
    val_semana_metrics = metricas_semana[1].to_dict(orient="dict")
    test_semana_metrics = metricas_semana[2].to_dict(orient="dict")

    # generates the residual plot
    residual_fig, res_list = plot_residual_error(
        df_target,
        pred_list,
        date_dataset_list,
    )
    residual_fig.savefig(os.path.join(VALUATION_PATH, "residuo.png"))
