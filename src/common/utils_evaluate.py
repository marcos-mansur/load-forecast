""" Module with utils to plot evaluation metrics """

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.seasonal import seasonal_decompose


def learning_curves(history, skip, plot=False):

    fig, ax = plt.subplots(figsize=(30, 6), nrows=1, ncols=3)
    metrics_list = [
        "loss",
        "val_loss",
        "mean_absolute_error",
        "val_mean_absolute_error",
        "root_mean_squared_error",
        "val_root_mean_squared_error",
    ]

    for i, metric in enumerate(metrics_list):
        # plot train and validation metrics on the same plot
        if i <= 1:
            ax1 = ax.ravel()[0]
        elif i > 1 and i <= 3:
            ax1 = ax.ravel()[1]
        else:
            ax1 = ax.ravel()[2]
        sns.lineplot(
            x=range(skip, len(history[metric])),
            y=history[metric][skip:],
            ax=ax1,
        )

    ax.ravel()[0].set_title("Learning Curve: MSE - loss")
    ax.ravel()[0].legend(labels=["Treino", "Validação"])
    ax.ravel()[1].set_title("Learning Curve: MAE")
    ax.ravel()[1].legend(labels=["Treino", "Validação"])
    ax.ravel()[2].set_title("Learning Curve: RMSE")
    ax.ravel()[2].legend(labels=["Treino", "Validação"])

    if plot:
        plt.show()

    return fig


def plot_predicted_series(pred_list, df_target, plot=False):

    colors = ["orange", "green", "purple"]
    fig, ax = plt.subplots(figsize=(20, 35), ncols=1, nrows=5)
    extra = plt.Rectangle((0, 0), 0, 0, fc="none", fill=False, ec="none", linewidth=0)

    # loop over 5 weeks
    for week_count in range(0, 5):
        # plot measured data
        sns.lineplot(
            x=df_target["Data"],
            y=df_target[f"Semana {week_count+1}"],
            ax=np.ravel(ax)[week_count],
            color="teal",
        )

        # plot predicted data
        for pred, color in zip(pred_list, colors[: len(pred_list)]):
            true_index = pred.index[week_count:]
            for _date_to_add_at_the_end in range(week_count + 1):
                np.append(
                    true_index,
                    (pd.Index([true_index[-1] + pd.Timedelta(value=7, unit="d")])),
                )
            y_value = pred.loc[:, f"previsão semana {week_count+1}"]
            sns.lineplot(
                x=[
                    item
                    for sublist in true_index.to_frame().values
                    for item in sublist
                ],
                y=y_value.values,
                ax=np.ravel(ax)[week_count],
                color=color,
            )

        np.ravel(ax)[week_count].set_title(
            f"Carga real vs Predição em todo o período - Semana {week_count+1}"
        )
        # np.ravel(ax)[week_count].legend(['Real','Previsão no treino',
        #     'Previsão na validação','Previsão no teste'], loc='upper left')
        score_list_by_dataset = []
        for pred_set in range(pred_list):

            # generate true date index
            true_index = pred.index[week_count:]
            for _date_to_add_at_the_end in range(week_count):
                true_index.append(
                    pd.Index([x_value[-1] + pd.Timedelta(value=7, unit="d")])
                )

            score_list_by_dataset.append(
                mean_squared_error(
                    pred_set.loc[:, f"previsão semana {week_count+1}"],
                    df_target[f"Semana {week_count+1}"].loc[true_index],
                    squared=False,
                )
            )

        scores = (
            r"MAE Train ={:.0f}"
            + "\n"
            + r"MAE val ={:.0f}"
            + "\n"
            + r"MAE test ={:.0f}"
        ).format(*score_list_by_dataset)
        np.ravel(ax)[week_count].legend([extra], [scores], loc="lower right")

    if plot:
        plt.show()
    return fig


def generate_metrics_semana(df_target, pred_list, date_list, plot=False):
    """Generates metrics for prediction performance of the 5 predicted weeks
    and generates plot for such metrics

    Args:
        df_target (_type_): dataframe with target values
        pred_list (_type_): list with predictions for train, val and test dataset as items
        date_list (_type_): list with week initial day data for train, val and test dataset as items
        id (_type_): _description_
        save (bool, optional): If True, saves plots as png. Defaults to False.
        plot (bool, optional): If True, show plots. Defaults to False.

    Returns:
        _type_: list with dataframes of
    """
    name_dict = {"0": "train_data", "1": "val_data", "2": "test_data"}

    fig, ax = plt.subplots(figsize=(20, 10), ncols=3, nrows=len(pred_list))

    # list of size equals to number of folds the dataset were splited
    enumerator = [cont for cont in range(len(pred_list))]

    train_metrics_df = pd.DataFrame(index=[f"Semana {week}" for week in range(1, 6)])
    val_metrics_df = pd.DataFrame(index=[f"Semana {week}" for week in range(1, 6)])
    test_metrics_df = pd.DataFrame(index=[f"Semana {week}" for week in range(1, 6)])
    metrics_df_list = [train_metrics_df, val_metrics_df, test_metrics_df]

    # df_set will take the values: train_pred, val_pred and test_pred
    for [i, df_set, data_week] in zip(enumerator, pred_list, date_list):

        mae_list = []
        mape_list = []
        mse_list = []

        # loops the five weeks
        for week in range(0, 5):

            # adds mae of each week to mae_list
            mae_list.append(
                mean_absolute_error(
                    df_set.loc[:, f"Semana {week+1}"],
                    df_target[f"Semana {week+1}"].loc[np.array(data_week.index)],
                )
            )

            # adds mape of each week to mape_list
            mape_list.append(
                mean_absolute_percentage_error(
                    df_set.loc[:, f"Semana {week+1}"],
                    df_target[f"Semana {week+1}"].loc[np.array(data_week.index)],
                )
                * 100
            )

            # adds mse of each week to mse_list
            mse_list.append(
                mean_squared_error(
                    df_set.loc[:, f"Semana {week+1}"],
                    df_target[f"Semana {week+1}"].loc[np.array(data_week.index)],
                    squared=False,
                )
            )

        # saves weekly metrics to a df
        metrics_df_list[i]["MAE"] = mae_list
        metrics_df_list[i]["MAPE"] = mape_list
        metrics_df_list[i]["RMSE"] = mse_list

        # rectangle to print the metrics mean and std over it
        extra = plt.Rectangle(
            (0, 0), 0, 0, fc="none", fill=False, ec="none", linewidth=0
        )

        # plot MAE by week
        sns.lineplot(x=range(1, 6), y=mae_list, ax=ax[i, 0])
        ax[i, 0].set_title(f"{name_dict[str(i)]} - MAE por semana prevista")
        ax[i, 0].set_xticks([1, 2, 3, 4, 5])
        scores = (r"$MAE={:.2f} \pm {:.2f}$").format(
            np.mean(mae_list), np.std(mae_list)
        )
        ax[i, 0].legend([extra], [scores], loc="lower right")

        # plot mape by week
        sns.lineplot(x=range(1, 6), y=mape_list, ax=ax[i, 1])
        ax[i, 1].set_title(f"{name_dict[str(i)]} - MAPE por semana prevista")
        ax[i, 1].set_xticks([1, 2, 3, 4, 5])
        scores = (r"$MAPE={:.2f} \pm {:.2f}$").format(
            np.mean(mape_list), np.std(mape_list)
        )
        ax[i, 1].legend([extra], [scores], loc="lower right")

        # plot MSE by week
        sns.lineplot(x=range(1, 6), y=mse_list, ax=ax[i, 2])
        ax[i, 2].set_title(f"{name_dict[str(i)]} - MSE por semana prevista")
        ax[i, 2].set_xticks([1, 2, 3, 4, 5])
        scores = (r"$MSE={:.2f} \pm {:.2f}$").format(
            np.mean(mse_list), np.std(mse_list)
        )
        ax[i, 2].legend([extra], [scores], loc="lower right")

    if plot:
        plt.show()
    return metrics_df_list, fig


def plot_sazonality(res_list, date_list, model="aditive"):
    assert model in ["aditive", "multiplicative"]

    analysis = [
        pd.Series(data=res_list[c], index=date_list[c]) for c in range(len(date_list))
    ]

    fig = []
    for i in range(len(date_list)):
        decompose_result_mult = seasonal_decompose(analysis[i], model=model)
        fig[i] = decompose_result_mult.plot()
        # fig[i].set_size_inches((20, 9))
    return fig


def create_target_df(df, df_target_path, baseline_size=1):
    """returns a dataframe with target values and baseline"""
    # average daily load by operative week
    df_target = pd.DataFrame(
        data=df.groupby(by=["semana"])["val_cargaenergiamwmed"].mean()
    )
    # start day of each operative week
    df_target.columns = ["Semana 1"]
    df_target["Semana 2"] = df_target["Semana 1"].shift(-1)
    df_target["Semana 3"] = df_target["Semana 1"].shift(-2)
    df_target["Semana 4"] = df_target["Semana 1"].shift(-3)
    df_target["Semana 5"] = df_target["Semana 1"].shift(-4)
    # defines the first day of Semana 1
    df_target["Data"] = df.groupby(by=["semana"])["din_instante"].min()
    df_target["dia semana"] = df.groupby(by=["semana"])["dia semana"].min()
    df_target["Resíduo"] = df_target["Semana 2"] - df_target["Semana 1"]
    df_target["Média Móvel"] = (
        df_target["Semana 1"].shift(1).rolling(baseline_size).mean()
    )

    df_target.to_csv(df_target_path)


def plot_residual_error(df_target, pred_list, date_list, plot=False):
    # variation from one week to the next
    res_baseline = df_target[
        "Resíduo"
    ]  # - df_target['Resíduo'].mean())/df_target['Resíduo'].max()
    # prediction residues
    colors = ["orange", "green", "purple"]

    fig, ax = plt.subplots(figsize=(20, 8))
    # diferença normalizada entre semanas consecutivas
    # sns.lineplot(y=res_baseline, x=df_target['Data'], ax=ax)

    res_list = []
    for pred, date, color in zip(pred_list, date_list, colors[: len(pred_list)]):

        res_pred = (
            pred.loc[:, "Semana 1"] - df_target["Semana 1"].loc[np.array(date.index)]
        )
        sns.lineplot(y=res_pred, x=df_target["Data"], ax=ax, color=color)
        res_list.append(res_pred)

    ax.set_title("Resíduo - Semana 1")
    ax.legend("")

    if plot:
        plt.show()
    return fig, res_list
