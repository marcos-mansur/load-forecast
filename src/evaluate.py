import json
import os

import tensorflow

from common.logger import get_logger
from const import *
from utils import *
from utils_tf import *


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


# valuation
if __name__ == "__main__":

    logger = get_logger(__name__)

    with open(HISTORY_PATH, "r") as history_file:
        history = json.load(history_file)
    
    load_dataset_list = load_featurized_data()
    date_dataset_list = load_featurized_week_data()
    
    params = yaml.safe_load(open("params.yaml"))

    model = tf.keras.models.load_model('/home/mamansur/projects/load-forecast/src/model/model_train.h5')

    # make prediction
    train_pred = predict_load(
        model,
        load_dataset_list["train_pred"],
        window_size=params["featurize"]["WINDOW_SIZE_PRO"],
    )
    val_pred = predict_load(
        model,
        load_dataset_list["val"],
        window_size=params["featurize"]["WINDOW_SIZE_PRO"],
    )
    test_pred = predict_load(
        model,
        load_dataset_list["test"],
        window_size=params["featurize"]["WINDOW_SIZE_PRO"],
    )
    logger.info("PREDICTIONS: DONE!")

    columns_names = [f'Semana {count}' for count in range(1,6)]
    if params["featurize"]["HOW_WINDOW_GEN_PRO"] == "autorregressivo":
        # unbatch
        pred_list = [
            pd.DataFrame(unbatch_pred(train_pred), columns=columns_names),
            pd.DataFrame(unbatch_pred(val_pred),columns=columns_names),
            pd.DataFrame(unbatch_pred(test_pred),columns=columns_names),
        ]
        logger.info("HOW_WINDOW_GEN_PRO = autoregressivo, predictions unbatched.")
    else:
        pred_list = [
            pd.DataFrame(train_pred,columns=columns_names),
            pd.DataFrame(val_pred,columns=columns_names),
            pd.DataFrame(test_pred,columns=columns_names),
        ]
        logger.info("HOW_WINDOW_GEN_PRO != autoregressivo, no need to unbatch predictions.")

    df_target = pd.read_csv(TARGET_DF_PATH)
    logger.info("LOADED TARGET DATA")
    os.makedirs(VALUATION_PATH, exist_ok=True)

    lc_fig = learning_curves(history=history, skip=20)
    lc_fig.savefig(os.path.join(VALUATION_PATH, "learning_curves.png"))
    logger.info("LEARNING CURVES SAVED TO DISK")

    # generates the plot of the original and
    # predicted time series for the 5 weeks
    pred_series_fig = plot_predicted_series(
        pred_list=pred_list,
        date_list=date_dataset_list,
        df_target=df_target,
    )
    pred_series_fig.savefig(os.path.join(VALUATION_PATH, "prediction_series.png"))
    logger.info("PREDICTED SERIES LINEPLOT SAVED TO DISK")

    # generates metrics for the 5 weeks and plots
    metricas_semana, metricas_fig = generate_metrics_semana(
        df_target, pred_list, date_dataset_list,
    )
    metricas_fig.savefig(os.path.join(VALUATION_PATH, "metrics_semana.png"))
    logger.info("WEEKLY METRICS GRAPHS SAVED TO DISK")

    train_semana_metrics = metricas_semana[0].to_dict(orient="dict")
    val_semana_metrics = metricas_semana[1].to_dict(orient="dict")
    test_semana_metrics = metricas_semana[2].to_dict(orient="dict")

    # generates the residual plot
    residual_fig, res_list = plot_residual_error(
        df_target, pred_list, date_dataset_list,
    )
    residual_fig.savefig(os.path.join(VALUATION_PATH, "residuo.png"))
    logger.info("RESIDUAL SAVED TO DISK")
