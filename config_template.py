# config.py template
# Add your respective folder names and save as config.py
# Please note that there are two roots used here, these can be the same root

# Root folder (EEG_age_prediction/)
ROOT = ""

# Second root for large data storage (e.g. External HDD)
SECOND_ROOT = ""

# Saved models
PATH_MODELS = ROOT + "trained_models/"

# Processed data for DL models
PATH_DATA_PROCESSED_DL = ROOT + "Data/data_processed_DL/"

# Processed data for ML models
PATH_DATA_PROCESSED_ML = ROOT + "Data/data_processed_ML/"

# EEG metadata folder
PATH_METADATA = ROOT + "Data/ePODIUM_metadata/"

# Raw EEG data folder
PATH_RAW_DATA = SECOND_ROOT + "Data/"

# Already preprocessed data for initial experiments - before using own preprocessing pipeline
PATH_DATA_PROCESSED_OLD = SECOND_ROOT + "Preprocessed_old/Data/data_processed_old/"