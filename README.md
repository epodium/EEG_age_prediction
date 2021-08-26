# EEG_age_prediction

Deep learning for age prediciton using EEG data.

## Structure of files


### Notebooks
The main content of this project can be found in the Notebooks/ folder. The following notebooks can be found here:

Initial experiments:
- Deep learning EEG_initial experiments.ipynb (This was used for initial testing of models, making use of an already preprocessed data set)

Preprocessing:
- Deep learning EEG_dataset preprocessing_DL.ipynb (Used for preprocessing the original EEG data set for DL purposes)
- Deep learning EEG_dataset preprocessing_ML.ipynb (Used for preprocessing the original EEG data set for ML purposes)
- Deep learning EEG_DL dataset_reduction.ipynb (Used for creating a reduced data set from DL data set)

Model training:
- Deep learning EEG_model testing_DL.ipynb (Training models on processed DL data set)
- Deep learning EEG_model testing_DL_REDUCED.ipynb (Training models on reduced processed DL data set)
- Deep learning EEG_model testing_ML.ipynb (Training models on processed ML data set)
- Deep learning EEG_Cross validation_Best DL.ipynb (Used for retraining the best model found in the other notebooks using a cross-validation approach) 

Model validation:
- Deep learning EEG_model validation_DL.ipynb (Validation/performance measures of DL models)
- Deep learning EEG_model validation_ML.ipynb (Validation/performance measures of ML models)

Weights inspection:
- Deep learning EEG_Model inspection.ipynb (Inspection of the weights of a DL model, visualizations)


### Configuration file

The config_template.py file should be renamed to config.py. Here the paths of the file locations can be stored. The ROOT folder can be the ROOT folder of this repository as well.

The Data folder contains the following folder/files:

- data_processed_DL/ (Folder with data generated using the DL preprocessing notebook)
- data_processed_ML/ (Folder with data generated using the ML preprocessing notebook)
- data_processed_DL/ (Folder with data generated using the DL dataset reduction notebook)
- ePODIUM_metadata/ (Folder containing metadata files)

We made use of a SECOND_ROOT, which was an external harddisk. On this harddisk the raw EEG data (.cnt files) was stored.


### Helper files

The main folder of this repository also contains a few helper files, for example DataGenerators.


## Data set

The data set of this project is not publicly available as it contains privacy-sensitive information.