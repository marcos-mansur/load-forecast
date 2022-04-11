# load-forecast

This repo contains the final project (currently under development) for my college degree as Bachelor of Production Engineering.

It consists of a Long Short-Term Memory Neural Network to predict daily average energy load of the next 5 weeks individually. The operative week for ONS (Nacional Operator of the Interconnected System) begins at Fridays and ends at Thursdays.  

## Summary

### Research Environment (Notebooks)
- [Data Quality and Preprocessing](https://github.com/marcos-mansur/load-forecast/blob/main/Notebooks/Data_quality.ipynb) - Check missing values and preprocess the data.
- [Baseline](https://github.com/marcos-mansur/load-forecast/blob/main/Notebooks/Notebook_8_Baseline.ipynb) - Codes for generating the baseline (daily avarege load of the week before)
- [Model_v1: First experimental model](https://github.com/marcos-mansur/load-forecast/blob/main/model_v1.ipynb) - Predicts the next week (7 days) daily average energy load. MAPE of ~3%, slightly better than the baseline.
- [Hyperparameter Tuning](https://github.com/marcos-mansur/load-forecast/blob/main/Notebooks/Notebook_9_Hyperparameter_tuning.ipynb) - Bayesian optimization structure to tune the following hyperparameters: the optmizer and it's parameters such as learning rate, momentum and decay; the amount of layers and neurons; kernel regularizer weight decay. Uses the lib optuna for the bayesian optimization implementation.
- [Model_v2: one shot prediction of five weeks](https://github.com/marcos-mansur/load-forecast/blob/main/model_v2.ipynb) - Outputs five values (single step) for each time series window input, each one the daily average load of one of the next five weeks.
