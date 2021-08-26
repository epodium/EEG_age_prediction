import sys, os, fnmatch, csv
import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn_rvm import EMRVR

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.utils import shuffle

sys.path.insert(0, os.path.dirname(os.getcwd()))

# Input and output folders
PATH_DATA_PROCESSED_ML= sys.argv[1]
PATH_OUTPUT = sys.argv[2]

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
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.fit_transform(X_val)
# X_test = scaler.fit_transform(X_test)

# MARK: reducing from 64 bit float to 32 bit float, to reduce memory usage
X_train = pd.DataFrame(scaler.fit_transform(X_train)).astype('float32')
X_val = pd.DataFrame(scaler.fit_transform(X_val)).astype('float32')
X_test = pd.DataFrame(scaler.fit_transform(X_test)).astype('float32')

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
codes_train_val = pd.concat([codes_train, codes_val])

del(file_names, files, df, frames, df_features, df_metadata, df_train, df_test, df_val, IDs_train, IDs_val, IDs_test, IDs_temp)
del(X_train, y_train, codes_train, X_val, y_val, codes_val)

# Shuffle data before using
X_train_val, y_train_val, codes_train_val = shuffle(X_train_val, y_train_val, codes_train_val, random_state=42)

chunked_X_train = np.array_split(X_train_val, 100)    
chunked_y_train = np.array_split(y_train_val, 100)

# chunks_X_train_ten = []
# chunks_y_train_ten = []

# for i in range(10):
#     chunks_X_train_ten.append(chunked_X_train[i])
#     chunks_y_train_ten.append(chunked_y_train[i])

# chunks_X_train_ten = pd.concat(chunks_X_train_ten)
# chunks_y_train_ten = pd.concat(chunks_y_train_ten)


# parameters = {'emrvr__kernel': ['poly', 'rbf', 'sigmoid'],
#               'emrvr__degree': [3, 4, 5, 6, 7],
#               'emrvr__epsilon': uniform(0, 6),
#               'emrvr__gamma': uniform(0.00001, 0.01)
# }

# scorer = make_scorer(mean_absolute_error, greater_is_better=False)
# pipe  = make_pipeline(StandardScaler(),
#                       EMRVR(verbose=True, max_iter=50000))

# RVR_randomsearch = RandomizedSearchCV(pipe, parameters, n_iter=100, 
#                                       cv=5, n_jobs=2, scoring=scorer, verbose=10)
# RVR_randomsearch.fit(chunked_X_train[0], chunked_y_train[0])

# output_file = os.path.join(PATH_OUTPUT, 'RVR_randomsearch.joblib')
# dump(RVR_randomsearch, output_file)

from sklearn_rvm import EMRVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMRVR = make_pipeline(StandardScaler(),
                      EMRVR(kernel='rbf', epsilon=1.5, gamma=(1/450)))

EMRVR.fit(chunked_X_train[0], chunked_y_train[0])

output_file = os.path.join(PATH_OUTPUT, 'EMRVR.joblib')
dump(EMRVR, output_file)