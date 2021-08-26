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

# ================ InceptionTime model ================ #

try:
    class Regressor_Inception:

        def __init__(self, output_directory, input_shape, verbose=False, build=True, batch_size=64,
                    nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

            self.output_directory = output_directory

            self.nb_filters = nb_filters
            self.use_residual = use_residual
            self.use_bottleneck = use_bottleneck
            self.depth = depth
            self.kernel_size = kernel_size - 1
            self.callbacks = None
            self.batch_size = batch_size
            self.bottleneck_size = 32
            self.nb_epochs = nb_epochs

            if build == True:
                self.model = self.build_model(input_shape)
                if (verbose == True):
                    self.model.summary()
                self.verbose = verbose
                self.model.save_weights(self.output_directory + '/inception_model_init.hdf5')

        def _inception_module(self, input_tensor, stride=1, activation='linear'):

            if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
                input_inception = tf.keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                    padding='same', activation=activation, use_bias=False)(input_tensor)
            else:
                input_inception = input_tensor

            # kernel_size_s = [3, 5, 8, 11, 17]
            kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

            conv_list = []

            for i in range(len(kernel_size_s)):
                conv_list.append(tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                    strides=stride, padding='same', activation=activation, use_bias=False)(
                    input_inception))

            max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

            conv_6 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                        padding='same', activation=activation, use_bias=False)(max_pool_1)

            conv_list.append(conv_6)

            x = tf.keras.layers.Concatenate(axis=2)(conv_list)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation='relu')(x)
            return x

        def _shortcut_layer(self, input_tensor, out_tensor):
            shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                            padding='same', use_bias=False)(input_tensor)
            shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

            x = tf.keras.layers.Add()([shortcut_y, out_tensor])
            x = tf.keras.layers.Activation('relu')(x)
            return x

        def build_model(self, input_shape):
            input_layer = tf.keras.layers.Input(input_shape)

            x = input_layer
            input_res = input_layer

            for d in range(self.depth):

                x = self._inception_module(x)

                if self.use_residual and d % 3 == 2:
                    x = self._shortcut_layer(input_res, x)
                    input_res = x

            pooling_layer = tf.keras.layers.AveragePooling1D(pool_size=50)(x)
            flat_layer = tf.keras.layers.Flatten()(pooling_layer)
            dense_layer = tf.keras.layers.Dense(128, activation='relu')(flat_layer)
            output_layer = tf.keras.layers.Dense(1)(dense_layer)

            model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

            return model

    model = Regressor_Inception(PATH_OUTPUT, input_shape, verbose=False).model

    optimizer = Adam(learning_rate=0.01)   
                
    model.compile(loss='mean_squared_error', 
                optimizer=optimizer, 
                metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    output_filename = f'Inception_regressor_{COUNT_MODEL}'
    output_file = os.path.join(PATH_OUTPUT, output_filename)

    checkpointer = ModelCheckpoint(filepath = output_file + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001, verbose=1)

    epochs = 1500

    # fit network
    print("LOGGING: Starting InceptionTime model training")
    history = model.fit(x=train_generator_noise,
                        validation_data=val_generator,
                        epochs=epochs,
                        verbose=2, 
                        max_queue_size=MAX_QUEUE_SIZE,
                        workers=WORKERS,  
                        callbacks = [checkpointer, earlystopper, reduce_lr])
    print("LOGGING: Finished InceptionTime model training")
except Exception as e:
    print("LOGGING: Failed InceptionTime model training:")
    print(e)
    pass