""" Module with functions to generate the target."""

import pandas as pd


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
    df_target.set_index("Data", inplace=True)
    df_target.dropna(subset=["Semana 5"], inplace=True, axis=0)
    df_target.to_csv(df_target_path)
