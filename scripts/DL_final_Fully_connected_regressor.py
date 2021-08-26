#!/usr/bin/env python

# ================ IMPORT LIBRARIES ================ #
import sys, os, fnmatch, time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.getcwd()))

from dataset_generator import DataGenerator

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, Input, Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, BatchNormalization, Dense, Conv1D, LeakyReLU, AveragePooling1D, Flatten, Reshape, MaxPooling1D
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

n_timesteps = 501
n_features = 30 
n_outputs = 1

COUNT_MODEL = "FINAL" # This will be appended to the saved model's name. To make sure to not overwrite models, increase this.
MAX_QUEUE_SIZE = 5000
WORKERS = 6

input_shape = (n_timesteps, n_features)

# Input and output folders
PATH_DATA_PROCESSED_DL = sys.argv[1]
PATH_OUTPUT = sys.argv[2]

# ================ INITIAL LOGS ================ #

print("LOGGING: Imported all modules")

# ================ LOAD PREPROCESSED DATA ================ #

# Step 1: Get all the files in the output folder
file_names = os.listdir(PATH_DATA_PROCESSED_DL)

# Step 2: Get the full paths of the files (without extensions)
files = [os.path.splitext(os.path.join(PATH_DATA_PROCESSED_DL, file_name))[0] for file_name in fnmatch.filter(file_names, "*.zarr")]

# Step 3: Load all the metadata
frames = []

for idx, feature_file in enumerate(files):
    df_metadata = pd.read_csv(feature_file.replace("processed_raw_", "processed_metadata_") + ".csv")
    frames.append(df_metadata)

df_metadata = pd.concat(frames) 

# Step 4: Add missing age information based on the age group the subject is in
df_metadata['age_months'].fillna(df_metadata['age_group'], inplace=True)
df_metadata['age_days'].fillna(df_metadata['age_group']*30, inplace=True)
df_metadata['age_years'].fillna(df_metadata['age_group']/12, inplace=True)

# Step 5: List all the unique subject IDs
subject_ids = sorted(list(set(df_metadata["code"].tolist())))

# Step 6: Split the subjects into train, val and test
IDs_train, IDs_temp = train_test_split(subject_ids, test_size=0.3, random_state=42)
IDs_test, IDs_val = train_test_split(IDs_temp, test_size=0.5, random_state=42)

# Step 7: Initialize DataGenerators
train_generator_noise = DataGenerator(list_IDs = IDs_train,
                                      BASE_PATH = PATH_DATA_PROCESSED_DL,
                                      metadata = df_metadata,
                                      n_average = 30,
                                      batch_size = 10,
                                      gaussian_noise=0.01,
                                      iter_per_epoch = 30,
                                      n_timepoints = 501, 
                                      n_channels=30, 
                                      shuffle=True)

val_generator = DataGenerator(list_IDs = IDs_val,
                              BASE_PATH = PATH_DATA_PROCESSED_DL,
                              metadata = df_metadata,
                              n_average = 30,
                              batch_size = 10,
                              iter_per_epoch = 100,
                              n_timepoints = 501,
                              n_channels=30,
                              shuffle=True)

print("LOGGING: Loaded all data and created generators")

# ================ Fully-connected neural network model ================ #

# Things to tweak: Dropout - increase?
# Add another layer?

try:
    def fully_connected_model():
        """ Returns the fully connected model from Ismail Fawaz et al. (2019). """

        input_layer = keras.layers.Input(input_shape)

        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(1)(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

    model = fully_connected_model()

    optimizer = Adadelta(learning_rate=0.01)    
                
    model.compile(loss='mean_squared_error', 
                optimizer=optimizer, 
                metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    output_filename = f'Fully_connected_regressor_{COUNT_MODEL}'
    output_file = os.path.join(PATH_OUTPUT, output_filename)

    checkpointer = ModelCheckpoint(filepath = output_file + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=1000, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, min_lr=0.0001, verbose=1)

    epochs = 5000

    print("LOGGING: Starting Fully-connected neural network model training")

    # fit network
    history = model.fit(x=train_generator_noise,
                        validation_data=val_generator,
                        epochs=epochs,
                        verbose=2,
                        max_queue_size=MAX_QUEUE_SIZE,
                        workers=WORKERS, 
                        callbacks=[checkpointer, earlystopper, reduce_lr])

    print("LOGGING: Finished Fully-connected neural network model training")
except Exception as e:
    print("LOGGING: Failed Fully-connected neural network model training:")
    print(e)
    pass