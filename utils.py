from builtins import print
import numpy as np
import pandas as pd
import matplotlib
import random

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import os
import operator
import utils

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def save_logs(output_directory, hist, y_pred, y_true, duration,
              lr=True, plot_test_acc=True):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    if plot_test_acc:
        df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    if plot_test_acc:
        df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    if plot_test_acc:
        # plot losses
        plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics

def calculate_metrics(y_true, y_pred, duration):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res

def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


    archive_name = ARCHIVE_NAMES[0]
    dataset_name = 'InlineSkate'
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    lengths = [2 ** i for i in range(5, 12)]

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    new_archive_name = 'InlineSkateXPs'

    for l in lengths:
        new_x_train = resample_dataset(x_train, l)
        new_x_test = resample_dataset(x_test, l)
        new_dataset_name = dataset_name + '-' + str(l)
        new_dataset_dir = root_dir + 'archives/' + new_archive_name + '/' + new_dataset_name + '/'
        create_directory(new_dataset_dir)

        np.save(new_dataset_dir + 'x_train.npy', new_x_train)
        np.save(new_dataset_dir + 'y_train.npy', y_train)
        np.save(new_dataset_dir + 'x_test.npy', new_x_test)
        np.save(new_dataset_dir + 'y_test.npy', y_test)