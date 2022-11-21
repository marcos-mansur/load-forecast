
import tensorflow as tf
import yaml
import pandas as pd
from src.common.logger import get_logger
from typing import Dict

import sys

logger = get_logger(__file__)
params = yaml.safe_load(open("params.yaml"))



def prepare_data_for_prediction(load_dataset_list, model_type):
    """Prepares data for training and prediction."""

    target_period = params["featurize"]["TARGET_PERIOD"]

    if model_type == "SINGLE-SHOT":
        return load_dataset_list["train"], load_dataset_list["val"]

    elif model_type == "AUTOREGRESSIVE":
        # convert pandas dataframe data to tf.data.Dataset so
        # the autoregressive model can operate
        train_x = tf.convert_to_tensor(load_dataset_list["train"][0])
        train_y = tf.convert_to_tensor(load_dataset_list["train"][1])
        train_concat = tf.concat(values=[train_x, train_y], axis=-1)
        train_data = tf.data.Dataset.from_tensor_slices(train_concat)
        train_data = train_data.map(
            lambda window: (
                window[:-target_period],
                window[-target_period:],
            )
        )
        train_data = train_data.batch(params["featurize"]["BATCH_SIZE_PRO"]).prefetch(1)

        val_x = tf.convert_to_tensor(load_dataset_list["val"][0])
        val_y = tf.convert_to_tensor(load_dataset_list["val"][1])
        val_concat = tf.concat(values=[val_x, val_y], axis=-1)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_concat)

        val_data = val_dataset.map(
            lambda window: (
                window[:-1],
                window[-params["featurize"]["TARGET_PERIOD"] :],
            )
        )
        val_data = val_data.batch(params["featurize"]["BATCH_SIZE_PRO"]).prefetch(1)

        return train_data, val_data
    
    else:
        logger.warning("[train.prepare_data] MODEL_TYPE NOT IDENTIFIED!")
        sys.exit(1)


def predict_autoregressive_load(model,data):
    
    predictions = model.predict(data)
    df_pred = pd.DataFrame(
        data={
            f"previsão semana {week+1}": predictions[:,week].reshape([-1]) 
            for week in range(5)
            }
        )
    return df_pred



def predict_load(model, input_and_target: pd.DataFrame, params: Dict):
    """ 
    If the param MODEL_TYPE is set to "AUTOREGRESSIVE" in params.yaml,
    return autoregressive predictions in array.

    If the param MODEL_TYPE is set to "SINGLE-SHOT" in params.yaml,
    return df with inputs and single-shot predictions.
       
    Args:
        model (_type_): _description_
        pred_dataset (_type_): _description_
        params (_type_): _description_

    Returns:
        _type_: _description_
    """
    autoreg_steps = params["featurize"]["TARGET_PERIOD"]
    model_type = params["featurize"]["MODEL_TYPE"]

    if model_type == "AUTOREGRESSIVE":
        predictions = predict_autoregressive_load(model, input_and_target)
        print("Forecasting type: AUTOREGRESSIVE... Done!")
        return predictions

    elif model_type == "SINGLE-SHOT":
        input_data = input_and_target[0]
        predictions = model.predict(input_data)
        inputs_and_predictions = input_data.merge(
            pd.DataFrame(
                predictions,
                index=input_data.index,
                columns=[
                    f"previsão semana {week}" for week in range(1, autoreg_steps + 1)
                ],
            ),
            on="din_instante",
            how="outer",
        )
        print("Forecasting type: SINGLE-SHOT... Done!")

        return inputs_and_predictions

def prepare_predicted_data(
    pred_data,
    load_dataset_list,
    model_type,
    target_period,
    window_size
):
    
    if model_type == 'AUTOREGRESSIVE':
        pred_data.index = load_dataset_list[0].index
        concat_pred = load_dataset_list[0].merge(
            pd.DataFrame(
                pred_data,
                index=load_dataset_list[0].index,
                columns=[
                    f"previsão semana {week}" for week in range(1, target_period + 1)
                ],
            ),
            on="din_instante",
            how="outer",
        )
        date_forecast = concat_pred.index + pd.Timedelta(value=7 * window_size, unit="d")
        concat_pred["Data Previsão"] = [str(date).split(" ")[0] for date in date_forecast]
        return concat_pred

    if model_type == 'SINGLE-SHOT':
        date_forecast = pred_data.index + pd.Timedelta(value=7 * window_size, unit="d")
        pred_data["Data Previsão"] = [str(date).split(" ")[0] for date in date_forecast]
        return pred_data
