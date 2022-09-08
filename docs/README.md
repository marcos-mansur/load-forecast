# Energy Load Forecast documentation


This repo contains the final project (currently under development) for my college degree as Bachelor of Production Engineering.

It consists of a Long Short-Term Memory Neural Network to predict average energy load of the next 5 weeks individually. 

The operative week for ONS (Nacional Operator of the Interconnected System) begins at Fridays and ends at Thursdays.  

## Reproduce

Fist, tou need to install the dependecies. You can do it with the following command on the root folder:
        
        pip install -r requirements/requirements.txt


This projects is structured using [DVC](https://dvc.org/doc). You can control the preprocess, featurize and model hyper-parameters through the [params.yaml file](https://dagshub.com/marcos-mansur/load-forecast/src/main/params.yaml). To run the worlflow, simply alter params.yaml at will and type into CLI from project root:

        dvc repro

DVC will check if the dependecies declared for each step of the pipeline in [dvc.yaml](https://dagshub.com/marcos-mansur/load-forecast/src/main/dvc.yaml) changed vs last run and re-run only the necessary steps. My experiments are tracked through MLFlow and registered in the "Experiments" tab in dagshub project repo. 

Validation visuals are stored to [valuation folder](https://dagshub.com/marcos-mansur/load-forecast/src/main/valuation).


## Files Summary

### Workflow

- [Preprocess.py](https://dagshub.com/marcos-mansur/load-forecast/src/main/src/preprocess.py) - Filters data by subsystem, impute missing energy load data points, guarantees that the data starts at a Friday and ends on a Thursday and parse dates to "Year" ("Ano"), "Month" ("Mês") and "Day" ("Dia") columns. Then, it splits data into train, validation and test sets.
- [Featurize.py](https://dagshub.com/marcos-mansur/load-forecast/src/main/src/featurize.py) - Apply transformations to turn data into time windows ready to be fed to model training and validation
- [Train.py](https://dagshub.com/marcos-mansur/load-forecast/src/main/src/train.py) - Trains the model and validates it.

### Suport
- [utils.py](https://dagshub.com/marcos-mansur/load-forecast/src/main/src/utils.py) - validation and suport functions.
- [const.py](https://dagshub.com/marcos-mansur/load-forecast/src/main/src/const.py) - Global constants.

### Pipeline

- [params.yaml](https://dagshub.com/marcos-mansur/load-forecast/src/main/params.yaml) - parameters controling preprocessing, featurizing and model hyper-parameters. 
- [dvc.yaml](https://dagshub.com/marcos-mansur/load-forecast/src/main/dvc.yamll) - defining dvc pipeline.
 
