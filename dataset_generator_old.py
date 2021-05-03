# Import libraries
from tensorflow.keras.utils import Sequence
import numpy as np
import os
import csv
import re # Hacky way to insert age now
from scipy import stats


class DataGenerator(Sequence):
    """Generates data for loading (preprocessed) EEG timeseries data.
    Create batches for training or prediction from given folders and filenames.

    """
    def __init__(self,
                 list_IDs,
                 main_labels,
                 ignore_labels,
                 filenames,
                 to_fit=True, #TODO: implement properly for False
                 gaussian_noise=0.0,
                 n_average = 30,
                 batch_size=32,
                 iter_per_epoch = 2,
                 up_sampling = True,
                 n_timepoints = 501,
                 n_channels=30,
                 include_baseline = False,
                 subtract_baseline = None,
                 shuffle=True,
                 warnings=False):
        """Initialization

        Args:
        --------
        list_IDs:
            list of all filename/label ids to use in the generator
        main_labels:
            list of all main labels.
        filenames:
            list of image filenames (file names)
        to_fit:
            True to return X and y, False to return X only
        n_average: int
            Number of EEG/time series epochs to average.
        batch_size:
            batch size at each iteration
        iter_per_epoch: int
            Number of iterations over all data points within one epoch.
        up_sampling: bool
            If true, create equal amounts of data for all main labels.
        n_timepoints: int
            Timepoint dimension of data.
        n_channels:
            number of input channels
        subtract_baseline: None, or list
            Give label (or list of labels) which should be used as a baseline.,
            All epochs of those labels will be averaged and then subtracted
            from all other epochs.
        shuffle:
            True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.main_labels = main_labels
        self.ignore_labels = ignore_labels
        self.filenames = filenames
        self.to_fit = to_fit
        self.gaussian_noise = gaussian_noise
        self.n_average = n_average
        self.batch_size = batch_size
        self.iter_per_epoch = iter_per_epoch
        self.up_sampling = up_sampling
        self.n_timepoints = n_timepoints
        self.n_channels = n_channels
        self.include_baseline = include_baseline
        self.subtract_baseline = subtract_baseline
        self.shuffle = shuffle
        self.warnings = warnings
        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch

        return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data

        Args:
        --------
        index: int
            index of the batch

        return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:((index + 1) * self.batch_size + 1)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[int(k)] for k in indexes]

        # Generate data
        X, y = self.generate_data(list_IDs_temp)

        if self.to_fit:
            return X, y
        else:
            return X


    def on_epoch_end(self):
        """Updates indexes after each epoch.
        Takes care of up-sampling and sampling frequency per "epoch".
        """

        idx_labels = []
        label_count = []

        # Up-sampling
        if self.up_sampling:
            main_labels = [self.main_labels[ID] for ID in self.list_IDs]
            labels_unique = list(set(main_labels))
            for label in labels_unique:
                idx = np.where(np.array(main_labels) == label)[0]
                idx_labels.append(idx)
                label_count.append(len(idx))

            idx_upsampled = np.zeros((0))
            for i in range(len(labels_unique)):
                up_sample_factor = self.iter_per_epoch * max(label_count)/label_count[i]
                idx_upsampled = np.concatenate((idx_upsampled, np.tile(idx_labels[i], int(up_sample_factor // 1))),
                                               axis = 0)
                idx_upsampled = np.concatenate((idx_upsampled, np.random.choice(idx_labels[i], int(label_count[i] * up_sample_factor % 1), replace=True)),
                                               axis = 0)
            self.indexes = idx_upsampled

        else:
            # No upsampling
            idx_base = np.arange(len(self.list_IDs))
            idx_epoch = np.tile(idx_base, self.iter_per_epoch)

            self.indexes = idx_epoch

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def get_all_data(self):
        # Generate data
        X, y = self.generate_data(self.list_IDs)
        return X, y.flatten()

    def generate_data(self, list_IDs_temp):
        """Generates data containing batch_size averaged time series.

        Args:
        -------
        list_IDs_temp: list
            list of label ids to load

        return: batch of averaged time series
        """
        X_data = np.zeros((0, self.n_channels, self.n_timepoints))
        y_data = []

        for i, ID in enumerate(list_IDs_temp):
            filename = self.filenames[ID]
            data_signal = self.load_signal(filename + '.npy')
            data_labels = self.load_labels(filename + '.csv')

            X, y = self.create_averaged_epoch(data_signal,
                                              data_labels)

            X_data = np.concatenate((X_data, X), axis=0)
            y_data += y

        if self.shuffle:
            idx = np.arange(len(y_data))
            np.random.shuffle(idx)
            X_data = X_data[idx, :, :]
            y_data = [y_data[i] for i in idx]

            
        return np.swapaxes(X_data,1,2), np.array(y_data).reshape((-1,1))
    
    def create_averaged_epoch(self,
                              data_signal,
                             data_labels):
        """
        Function to create averages of self.n_average epochs.
        Will create one averaged epoch per found unique label from self.n_average random epochs.

        Args:
        --------
        data_signal: numpy array
            Data from one person as numpy array
        data_labels: list
            List of labels for all data from one person.
        """
        # Create new data collection:
        X_data = np.zeros((0, self.n_channels, self.n_timepoints))
        y_data = []


        categories_found = list(set(data_labels))
        if self.subtract_baseline is None:
            self.subtract_baseline = []
        baseline_labels_found = list(set(categories_found) & set(self.subtract_baseline))
        
        if len(baseline_labels_found) > 0:
            idx_baseline = []
            for cat in baseline_labels_found:
                # Create baseline to subtract from all other epochs.
                idx_baseline.extend(np.where(np.array(data_labels) == cat)[0])

            baseline = np.mean(data_signal[idx_baseline,:,:], axis=0)
        else:
            baseline = 0

        idx_cat = []
        for cat in list(set(categories_found) - set(self.subtract_baseline)):
            if cat not in self.ignore_labels:
                idx = np.where(np.array(data_labels) == cat)[0]
                idx_cat.append(idx)
                
                if len(idx) >= self.n_average:
                    select = np.random.choice(idx, self.n_average, replace=False)
                else:
                    if self.warnings:
                        print("Found only", len(idx), " epochs and will take those!")
                    signal_averaged = np.mean(data_signal[idx,:,:], axis=0)
                    break
                    
                # Average signal and subtract baseline (if any)
                signal_averaged = np.mean(data_signal[select,:,:], axis=0) - baseline
                X_data = np.concatenate([X_data, np.expand_dims(signal_averaged, axis=0)], axis=0)
                y_data.append(cat)
            else:
                pass
        
        if self.gaussian_noise != 0.0:
            X_data += np.random.normal(0, self.gaussian_noise, X_data.shape)

        return X_data, y_data


    def load_signal(self,
                    filename):
        """Load EEG signal from one person.

        Args:
        -------
        filename: str
            filename...

        return: loaded array
        """
        return np.load(os.path.join(filename))


    def load_labels(self,
                    filename):
        metadata = []
        filename = os.path.join(filename)
        with open(filename, 'r') as readFile:
            reader = csv.reader(readFile, delimiter=',')
            for row in reader:
                #if len(row) > 0:
                metadata.append(row)
        readFile.close()
        
        age_in_months = re.findall(r'\d+', filename)[0] # Hacky way to insert age now
        age_labels = [float(age_in_months) for label in metadata[0]] # Hacky way to insert age now
        
        return age_labels