import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


def learning_curves(history, skip, id, save=False, plot=False):

  fig,ax = plt.subplots(figsize=(30,6), nrows=1, ncols=3)
  metrics_list = ['loss',
                  'val_loss',
                  'mean_absolute_error', 
                  'val_mean_absolute_error',
                  'root_mean_squared_error',
                  'val_root_mean_squared_error'
                  ]

  for i, metric in enumerate(metrics_list):
    # plot train and validation metrics on the same plot
    if i<=1:
      ax1 =  ax.ravel()[0]
    elif i>1 and i<=3:
      ax1 =  ax.ravel()[1]
    else:
      ax1= ax.ravel()[2]
    sns.lineplot(x = range(skip,len(history.history[metric])),
                y = history.history[metric][skip:],
                ax = ax1)


  ax.ravel()[0].set_title("Learning Curve: MSE - loss")
  ax.ravel()[0].legend(labels=['Treino', 'Validação'])
  ax.ravel()[1].set_title("Learning Curve: MAE")
  ax.ravel()[1].legend(labels=['Treino', 'Validação'])
  ax.ravel()[2].set_title("Learning Curve: RMSE")
  ax.ravel()[2].legend(labels=['Treino', 'Validação'])

  if save:
    fig.savefig(f"valuation/learning_curves{id}.png")
  if plot:
    plt.show()


def plot_pred(  date_list, pred_list, 
                df_target, id, 
                baseline=False, 
                save=False, plot=False):

    colors = ['orange', 'green', 'purple']
    _,ax=plt.subplots(figsize=(20,35), ncols=1, nrows=5)
    extra = plt.Rectangle((0, 0), 0, 0, fc="none",
                            fill=False, ec='none', linewidth=0)

    # plot baseline
    if baseline == True:
        sns.lineplot(x = df_target['Data'],
                y = df_target['Média Móvel'], 
                ax=np.ravel(ax)[0],
                color='black')


    for x in range(0,5):
        # plot measured data
        sns.lineplot(x = df_target['Data'], 
                    y = df_target[f'Semana {x+1}'], 
                    ax=np.ravel(ax)[x], 
                    color = 'teal')

        # plot predicted data
        for date,pred,color in zip(date_list, 
                                pred_list,
                                colors[:len(pred_list)]): 
            sns.lineplot(x = date.shift(x).din_instante,
                        y = pred[:,x], 
                        ax=np.ravel(ax)[x],
                        color=color)
                        

        np.ravel(ax)[x].set_title(f'Carga real vs Predição em todo o período - Semana {x+1}')
        # np.ravel(ax)[x].legend(['Real','Previsão no treino','Previsão na validação','Previsão no teste'], loc='upper left')

        score = [tf.keras.metrics.mean_absolute_error(df_target[f'Semana {x+1}'].loc[np.array(date_list[j].index)],
                                                    pred_list[j][:,x]).numpy() for j in range(len(pred_list))]
        scores = (r"MAE Train ={:.0f}"+'\n'+r"MAE val ={:.0f}"+"\n"+r"MAE test ={:.0f}").format(
                *score
                # tf.keras.metrics.mean_absolute_error(df_target[f'Semana {x+1}'].loc[np.array(date_list[0].index)],
                #                                     pred_list[0][:,x]).numpy(),
                # tf.keras.metrics.mean_absolute_error(df_target[f'Semana {x+1}'].loc[np.array(date_list[1].index)],
                #                                     pred_list[1][:,x]).numpy(),
                # tf.keras.metrics.mean_absolute_error(df_target[f'Semana {x+1}'].loc[np.array(date_list[2].index)],
                #                                     pred_list[2][:,x]).numpy() 
                )
        np.ravel(ax)[x].legend([extra], [scores], loc='lower right')
    if save:
        plt.savefig(f"valuation/prediction_series{id}.png")
    if plot:
      plt.show()


def metrics_semana(df_target, pred_list,date_list,id, save=False, plot=False):
    name_dict = {'0':'train_data',
                '1':'val_data',
                '2': 'test_data'}

    fig, ax = plt.subplots(figsize=(20,10),ncols=3,nrows=len(pred_list))

    # list of size equals to number of folds the dataset were splited
    contador = [cont for cont in range(len(pred_list))]

    for [i,df_loop,data_week] in zip(contador,pred_list, date_list):

        mae_list = []
        mape_list = []
        mse_list = []

        for x in range(0,5):
            #print(f"MAE train_pred Semana {i+1}:  {mean_absolute_error(train_pred[:,i], df_target[5 : int(len(df_target)*0.7)-4].iloc[:,i])}")
            mae_list.append(mean_absolute_error(df_loop[:,x], 
                                                df_target[f'Semana {x+1}'].loc[np.array(data_week.index)]))
        for x in range(0,5):
            #print(f"MAPE train_pred Semana {i+1}: {mean_absolute_percentage_error(train_pred[:,i], df_target[5 : int(len(df_target)*0.7)-4].iloc[:,i])*100}")
            mape_list.append(mean_absolute_percentage_error(df_loop[:,x], 
                                                            df_target[f'Semana {x+1}'].loc[np.array(data_week.index)])*100)
        for x in range(0,5):
            #print(f"MSE train_pred Semeana {i+1}: {mean_squared_error(train_pred[:,i], df_target[5 : int(len(df_target)*0.7)-4].iloc[:,i])}")                
            mse_list.append(mean_squared_error(df_loop[:,x], df_target[f'Semana {x+1}'].loc[np.array(data_week.index)]))
        

        # rectangle to print the metrics mean and std over it 
        extra = plt.Rectangle((0, 0), 0, 0, fc="none", fill=False, ec='none', linewidth=0)
        # plot MAE by week
        sns.lineplot(x=range(1,6),y=mae_list, ax=ax[i,0])
        ax[i,0].set_title(f'{name_dict[str(i)]} - MAE por semana prevista')
        ax[i,0].set_xticks([1,2,3,4,5])
        scores = (r'$MAE={:.2f} \pm {:.2f}$').format(np.mean(mae_list),np.std(mae_list))
        ax[i,0].legend([extra], [scores], loc='lower right')
        # plot mape by week
        sns.lineplot(x=range(1,6),y=mape_list, ax=ax[i,1])
        ax[i,1].set_title(f'{name_dict[str(i)]} - MAPE por semana prevista')
        ax[i,1].set_xticks([1,2,3,4,5])
        scores = (r'$MAPE={:.2f} \pm {:.2f}$').format(np.mean(mape_list),np.std(mape_list))
        ax[i,1].legend([extra], [scores], loc='lower right')
        # plot MSE by week
        sns.lineplot(x=range(1,6),y=mse_list, ax=ax[i,2])
        ax[i,2].set_title(f'{name_dict[str(i)]} - MSE por semana prevista')
        ax[i,2].set_xticks([1,2,3,4,5])
        scores = (r'$MSE={:.2f} \pm {:.2f}$').format(np.mean(mse_list),np.std(mse_list))
        ax[i,2].legend([extra], [scores], loc='lower right')

    if save:
        fig.savefig(f"valuation/metrics_semana{id}.png")
    if plot:
      plt.show()


def sazonality(df,id=1,model='aditive',save=False):
    assert model in ['aditive', 'multiplicative']
    mean_load_week = df.groupby(by=['semana'])['val_cargaenergiamwmed'].mean()  
    date_week = df.groupby(by=['semana'])['din_instante'].min()
    df_sazo = pd.DataFrame(data=mean_load_week)
    # data de inicio da semana prevista
    df_sazo['din_instante'] = date_week

    analysis = df_sazo.set_index('din_instante')[['val_cargaenergiamwmed']].copy()

    decompose_result_mult = seasonal_decompose(analysis, model=model)

    fig = decompose_result_mult.plot();
    fig.set_size_inches((20, 9))
    if save:
        fig.savefig(f'valuation/sazonalidade_{id}_{model}.png')


def create_target_df(df, df_target_path, baseline_size=1):
    """ returns a dataframe with target values and baseline"""
    # average daily load by operative week
    df_target = pd.DataFrame(data=df.groupby(by=['semana'])['val_cargaenergiamwmed'].mean())
    # start day of each operative week
    df_target.columns = ['Semana 1']
    df_target['Semana 2'] = df_target['Semana 1'].shift(-1)
    df_target['Semana 3'] = df_target['Semana 1'].shift(-2)
    df_target['Semana 4'] = df_target['Semana 1'].shift(-3)
    df_target['Semana 5'] = df_target['Semana 1'].shift(-4)
    # defines the first day of Semana 1
    df_target['Data'] = df.groupby(by=['semana'])['din_instante'].min()
    df_target['dia semana'] = df.groupby(by=['semana'])['dia semana'].min()
    df_target['Resíduo'] = df_target['Semana 2']  - df_target['Semana 1']
    df_target['Média Móvel'] = df_target['Semana 1'].shift(1).rolling(baseline_size).mean()

    df_target.to_csv(df_target_path)


def plot_res(df_target,pred_list, date_list,id, save=False, plot=False):
    # variation from one week to the next
    res_baseline = df_target['Resíduo'] #- df_target['Resíduo'].mean())/df_target['Resíduo'].max()
    # prediction residues
    colors = ['orange', 'green', 'purple']


    fig, ax =plt.subplots(figsize=(20,8))
    # diferença normalizada entre semanas consecutivas
    #sns.lineplot(y=res_baseline, x=df_target['Data'], ax=ax)

    for pred, date, color in zip(pred_list,date_list,colors[:len(pred_list)]):
        
        res_pred = pred[:,0] - df_target[f'Semana 1'].loc[np.array(date.index)]
        sns.lineplot(y=res_pred, x=df_target['Data'], ax=ax, color= color)

    ax.set_title("Resíduo - Semana 1")
    ax.legend('')
    if save:
        fig.savefig(f"valuation/residuo{id}.png")
    if plot:
      plt.show()