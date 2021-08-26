#!/usr/bin/env python

# ================ IMPORT LIBRARIES ================ #
import sys, os, fnmatch, time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.getcwd()))

from dataset_generator_reduced import DataGeneratorReduced

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

COUNT_MODEL = "04" # This will be appended to the saved model's name. To make sure to not overwrite models, increase this.
MAX_QUEUE_SIZE = 5000
WORKERS = 6

input_shape = (n_timesteps, n_features)

# Input and output folders
PATH_DATA_PROCESSED_DL_REDUCED = sys.argv[1]
PATH_OUTPUT = sys.argv[2]

# ================ INITIAL LOGS ================ #

print("LOGGING: Imported all modules")

# ================ LOAD PREPROCESSED DATA ================ #

# Step 1: Get all the files in the output folder
file_names = os.listdir(PATH_DATA_PROCESSED_DL_REDUCED)

# Step 2: Get the full paths of the files (without extensions)
files = [os.path.splitext(os.path.join(PATH_DATA_PROCESSED_DL_REDUCED, file_name))[0] for file_name in fnmatch.filter(file_names, "*.zarr")]

# Step 3: Load all the metadata
frames = []

for idx, feature_file in enumerate(files):
    df_metadata = pd.read_csv(feature_file + ".csv")
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
train_generator_noise = DataGeneratorReduced(list_IDs = IDs_train,
                                             BASE_PATH = PATH_DATA_PROCESSED_DL_REDUCED,
                                             metadata = df_metadata,
                                             n_average = 3,
                                             batch_size = 10,
                                             gaussian_noise=0.01,
                                             iter_per_epoch = 30,
                                             n_timepoints = 501, 
                                             n_channels=30, 
                                             shuffle=True)

val_generator = DataGeneratorReduced(list_IDs = IDs_val,
                                     BASE_PATH = PATH_DATA_PROCESSED_DL_REDUCED,
                                     metadata = df_metadata,
                                     n_average = 3,
                                     batch_size = 10,
                                     n_timepoints = 501,
                                     iter_per_epoch = 100,
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

        layer_4 = keras.layers.Dropout(0.2)(layer_3)
        layer_4 = keras.layers.Dense(500, activation='relu')(layer_4)

        output_layer = keras.layers.Dropout(0.3)(layer_4)
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

# ================ Convolutional neural network model ================ #

try:
    def cnn_model():
        """ Returns the CNN (FCN) model from Ismail Fawaz et al. (2019). """

        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=256, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=512, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(1)(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model 

    model = cnn_model()

    optimizer = Adam(learning_rate=0.01)    
                
    model.compile(loss='mean_squared_error', 
                optimizer=optimizer, 
                metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    output_filename = f'CNN_regressor_{COUNT_MODEL}'
    output_file = os.path.join(PATH_OUTPUT, output_filename)

    checkpointer = ModelCheckpoint(filepath = output_file + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=250, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001, verbose=1)

    epochs = 2000

    print("LOGGING: Starting Convolutional neural network model training")

    # fit network
    history = model.fit(x=train_generator_noise,
                        validation_data=val_generator,
                        epochs=epochs, 
                        verbose=2, 
                        max_queue_size=MAX_QUEUE_SIZE,
                        workers=WORKERS, 
                        callbacks=[checkpointer, earlystopper, reduce_lr])
    
    print("LOGGING: Finished Convolutional neural network model training")
except Exception as e:
    print("LOGGING: Failed Convolutional neural network model training:")
    print(e)
    pass

# ================ ResNet model ================ #

try:
    def resnet_model():
        """ Returns the ResNet model from Ismail Fawaz et al. (2019). """
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=16, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=10, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=6, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=16, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=10, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=6, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=16, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=10, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=6, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(1)(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

    model = resnet_model()

    optimizer = Adam(learning_rate=0.01)    
                
    model.compile(loss='mean_squared_error', 
                optimizer=optimizer, 
                metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    output_filename = f'ResNet_regressor_{COUNT_MODEL}'
    output_file = os.path.join(PATH_OUTPUT, output_filename)

    checkpointer = ModelCheckpoint(filepath = output_file + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=250, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001, verbose=1)

    epochs = 1500

    print("LOGGING: Starting ResNet model training")

    # fit network
    history = model.fit(x=train_generator_noise,
                        validation_data=val_generator,
                        epochs=epochs, 
                        verbose=2, 
                        max_queue_size=MAX_QUEUE_SIZE,
                        workers=WORKERS,  
                        callbacks=[checkpointer, earlystopper, reduce_lr])
    print("LOGGING: Finished ResNet model training")
except Exception as e:
    print("LOGGING: Failed ResNet model training:")
    print(e)
    pass

# ================ Encoder model ================ #

try:
    def encoder_model():
        """ Returns the Encoder model from Ismail Fawaz et al. (2019). """
        input_layer = keras.layers.Input(input_shape)

        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(input_layer)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:,:,:256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:,:,256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax,attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=256,activation='sigmoid')(multiply_layer)
        dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(1)(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

    model = encoder_model()

    optimizer = Adam(learning_rate=0.01)    
                
    model.compile(loss='mean_squared_error', 
                optimizer=optimizer, 
                metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    output_filename = f'Encoder_regressor_{COUNT_MODEL}'
    output_file = os.path.join(PATH_OUTPUT, output_filename)

    checkpointer = ModelCheckpoint(filepath = output_file + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.00001, verbose=1)

    epochs = 500

    print("LOGGING: Starting Encoder model training")
    # fit network
    history = model.fit(x=train_generator_noise,
                        validation_data=val_generator,
                        epochs=epochs, 
                        verbose=2, 
                        max_queue_size=MAX_QUEUE_SIZE,
                        workers=WORKERS,  
                        callbacks=[checkpointer, reduce_lr])
    print("LOGGING: Finished Encoder model training")
except Exception as e:
    print("LOGGING: Failed Encoder model training:")
    print(e)
    pass

# ================ Time-CNN model ================ #

try:
    def timecnn_model():
        """ Returns the Time-CNN model from Ismail Fawaz et al. (2019). """
        
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        conv3 = keras.layers.Conv1D(filters=24,kernel_size=7,padding=padding,activation='sigmoid')(conv2)
        conv3 = keras.layers.AveragePooling1D(pool_size=3)(conv3)

        flatten_layer = keras.layers.Flatten()(conv3)

        output_layer = keras.layers.Dense(units=1)(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

    model = timecnn_model()

    optimizer = Adam(learning_rate=0.01)    
                
    model.compile(loss='mean_squared_error', 
                optimizer=optimizer, 
                metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    output_filename = f'TimeCNN_regressor_{COUNT_MODEL}'
    output_file = os.path.join(PATH_OUTPUT, output_filename)

    checkpointer = ModelCheckpoint(filepath = output_file + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=250, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001, verbose=1)

    epochs = 2000

    print("LOGGING: Starting Time-CNN model training")

    # fit network
    history = model.fit(x=train_generator_noise,
                        validation_data=val_generator,
                        epochs=epochs, 
                        verbose=2, 
                        max_queue_size=MAX_QUEUE_SIZE,
                        workers=WORKERS,  
                        callbacks=[checkpointer, earlystopper, reduce_lr])
    print("LOGGING: Finished Time-CNN model training")
except Exception as e:
    print("LOGGING: Failed Time-CNN model training:")
    print(e)
    pass

# ================ BLSTM-LSTM model ================ #

try:
    def blstm_lstm_model():
        """ Returns the BLSTM-LSTM model from Kaushik et al. (2019). """
        
        # MARK: This model compresses too much in the last phase, check if possible to improve.
        
        model = keras.Sequential()
        
        # BLSTM layer
        model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(.2))
        model.add(BatchNormalization())
        
        # LSTM layer
        model.add(LSTM(128, return_sequences=True))
        model.add(BatchNormalization())

        # LSTM layer
        model.add(LSTM(64, return_sequences=True))
        model.add(BatchNormalization())

        # LSTM layer
        model.add(LSTM(32, return_sequences=False))
        model.add(BatchNormalization())
        
        # Fully connected layer
        model.add(Dense(32))
        
        model.add(Dense(n_outputs))
        
        return model 

    model = blstm_lstm_model()

    optimizer = Adam(learning_rate=0.01)    
                
    model.compile(loss='mean_squared_error', 
                optimizer=optimizer, 
                metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    output_filename = f'BLSTM_regressor_{COUNT_MODEL}'
    output_file = os.path.join(PATH_OUTPUT, output_filename)

    checkpointer = ModelCheckpoint(filepath = output_file + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=250, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001, verbose=1)

    epochs = 1500

    print("LOGGING: Starting BLSTM-LSTM model training")
    # fit network
    history = model.fit(x=train_generator_noise,
                        validation_data=val_generator,
                        epochs=epochs, 
                        verbose=2, 
                        max_queue_size=MAX_QUEUE_SIZE,
                        workers=WORKERS,  
                        callbacks=[checkpointer, earlystopper, reduce_lr])
    print("LOGGING: Finished BLSTM-LSTM model training")
except Exception as e:
    print("LOGGING: Failed BLSTM-LSTM model training:")
    print(e)
    pass

# ================ InceptionTime model ================ #

try:
    class Regressor_Inception:

        def __init__(self, output_directory, input_shape, verbose=False, build=True, batch_size=64,
                    nb_filters=32, use_residual=True, use_bottleneck=True, depth=9, kernel_size=41, nb_epochs=1500):

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