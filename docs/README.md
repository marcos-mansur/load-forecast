# Energy Load Forecast documentation
![https://github.com/psf/black](https://img.shields.io/badge/code%20style-black-000000.svg)

This repo contains the final project (currently under development) for my college degree as Bachelor of Production Engineering.

It consists of a Long Short-Term Memory Neural Network to predict average energy load of the next 5 weeks individually. 

The operative week for ONS (Nacional Operator of the Interconnected System) begins at Fridays and ends at Thursdays.  

## Reproduce

Fisrt, tou need to install the dependecies. You can do it with the following command on the root folder:
        
        pip install -r requirements/requirements.txt


This projects is structured using [DVC](https://dvc.org/doc). You can control the preprocess, featurize and model hyper-parameters through the [params.yaml file](https://github.com/marcos-mansur/load-forecast/blob/main/params.yaml). To run the worlflow, simply alter params.yaml at will and type into CLI from project root:

        dvc repro

DVC will check if the dependecies declared for each step of the pipeline in [dvc.yaml](https://github.com/marcos-mansur/load-forecast/blob/main/dvc.yaml) changed vs last run and re-run only the necessary steps. My experiments are tracked through MLFlow and registered in the "Experiments" tab in dagshub project repo. 

Validation visuals are stored to [valuation folder](https://github.com/marcos-mansur/load-forecast/tree/main/evaluation).
