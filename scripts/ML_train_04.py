import sys, os, fnmatch, csv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

sys.path.insert(0, os.path.dirname(os.getcwd()))

# Input and output folders
PATH_DATA_PROCESSED_ML= sys.argv[1]
PATH_OUTPUT = sys.argv[2]

MAX_QUEUE_SIZE = 5000
WORKERS = 6

# Step 1: Get all the files in the output folder
file_names = os.listdir(PATH_DATA_PROCESSED_ML)

# Step 2: Get the full paths of the files (without extensions)
files = [os.path.splitext(os.path.join(PATH_DATA_PROCESSED_ML, file_name))[0] for file_name in fnmatch.filter(file_names, "*.h5")]

# Step 3: Load the features
frames = []

for idx, feature_file in enumerate(files):
    df_features = pd.read_hdf(feature_file + ".h5")
    df_metadata = pd.read_csv(feature_file.replace("extracted_features_", "processed_data_") + ".csv")
    
    # Step 4: Assign labels
    df_features['label'] = df_metadata['age_months'][0]
    
    # Step 5: Assign subject code
    df_features['code'] = df_metadata['code'][0]
    frames.append(df_features)

df = pd.concat(frames) 

# Step 6: List all the unique subject IDs
subject_ids = sorted(list(set(df["code"].tolist())))

IDs_train, IDs_temp = train_test_split(subject_ids, test_size=0.3, random_state=42)
IDs_test, IDs_val = train_test_split(IDs_temp, test_size=0.5, random_state=42)

# Step 7: Split the DataFrames into train, validation and test
df_train = df[df['code'].isin(IDs_train)]
df_val = df[df['code'].isin(IDs_val)]
df_test = df[df['code'].isin(IDs_test)]

feature_names = df.columns.values

X_train = df_train.drop(['label', 'code'], axis=1).reset_index(drop=True)
y_train = df_train['label'].reset_index(drop=True)
codes_train = df_train['code'].reset_index(drop=True)

X_val = df_val.drop(['label', 'code'], axis=1).reset_index(drop=True)
y_val = df_val['label'].reset_index(drop=True)
codes_val = df_val['code'].reset_index(drop=True)

X_test = df_test.drop(['label', 'code'], axis=1).reset_index(drop=True)
y_test = df_test['label'].reset_index(drop=True)
codes_test = df_test['code'].reset_index(drop=True)

scaler = StandardScaler()

# MARK: reducing from 64 bit float to 32 bit float, to reduce memory usage
X_train = pd.DataFrame(scaler.fit_transform(X_train)).astype('float32')
X_val = pd.DataFrame(scaler.fit_transform(X_val)).astype('float32')
X_test = pd.DataFrame(scaler.fit_transform(X_test)).astype('float32')

del(file_names, files, df, frames, df_features, df_metadata, df_train, df_test, df_val, IDs_train, IDs_val, IDs_test, IDs_temp)

input_shape=(450, )

try:
    def fully_connected_model():
        model = keras.Sequential()
        
        model.add(Dense(512, activation='tanh', input_shape=input_shape))
        model.add(BatchNormalization())
            
        model.add(Dense(128, activation='tanh'))
        model.add(BatchNormalization())

        model.add(Dense(1))
        
        return model
    
    model = fully_connected_model()

    optimizer = Adadelta(learning_rate=0.01)    
    model.compile(loss='mean_squared_error', 
            optimizer=optimizer, 
            metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    output_filename = 'FC_regressor_04'
    output_file = os.path.join(PATH_OUTPUT, output_filename)

    checkpointer = ModelCheckpoint(filepath = output_file + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=1000, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001, verbose=1)

    epochs = 5000

    print("LOGGING: Starting FC_regressor_04 training")

    # fit network
    history = model.fit(x=X_train,
                        y=y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        verbose=2,
                        max_queue_size=MAX_QUEUE_SIZE,
                        workers=WORKERS, 
                        callbacks = [checkpointer, earlystopper, reduce_lr])
except Exception as e:
    print("LOGGING: Failed FC_regressor_04 training:")
    print(e)
    pass
