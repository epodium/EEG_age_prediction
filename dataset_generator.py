# Import libraries
from tensorflow.keras.utils import Sequence
import numpy as np
import zarr
import os

class DataGenerator(Sequence):
    """Generates data for loading (preprocessed) EEG timeseries data.
    Create batches for training or prediction from given folders and filenames.

    """
    def __init__(self,
                 list_IDs,
                 BASE_PATH,
                 metadata,
                 gaussian_noise=0.0,
                 n_average = 30,
                 batch_size=32,
                 iter_per_epoch = 1,
                 n_timepoints = 501,
                 n_channels=30,
                 shuffle=True,
                 warnings=False):
        """Initialization

        Args:
        --------
        list_IDs:
            list of all filename/label ids to use in the generator
        metadata:
            DataFrame containing all the metadata.
        n_average: int
            Number of EEG/time series epochs to average.
        batch_size:
            batch size at each iteration
        iter_per_epoch: int
            Number of iterations over all data points within one epoch.
        n_timepoints: int
            Timepoint dimension of data.
        n_channels:
            number of input channels
        shuffle:
            True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.BASE_PATH = BASE_PATH
        self.metadata = metadata
        self.metadata_temp = None
        self.gaussian_noise = gaussian_noise
        self.n_average = n_average
        self.batch_size = batch_size
        self.iter_per_epoch = iter_per_epoch
        self.n_timepoints = n_timepoints
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.warnings = warnings
        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch

        return: number of batches per epoch
        """
        return int(np.floor(len(self.metadata_temp) / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data

        Args:
        --------
        index: int
            index of the batch

        return: X and y when fitting. X only when predicting
        """
        
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:((index + 1) * self.batch_size)]

        # Get temporary metadata, based on the indices of the batch
        temporary_metadata = self.metadata_temp.iloc[indexes]

        # Generate data
        X, y = self.generate_data(temporary_metadata)

        return X, y


    def on_epoch_end(self):
        """Updates indexes after each epoch."""

        # Create new metadata DataFrame with only the current subject IDs
        if self.metadata_temp is None:
            self.metadata_temp = self.metadata[self.metadata['code'].isin(self.list_IDs)].reset_index(drop=True)
                               
        idx_base = np.arange(len(self.metadata_temp))
        idx_epoch = np.tile(idx_base, self.iter_per_epoch)

        self.indexes = idx_epoch

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def get_all_data(self):
        # Generate data
        X, y = self.generate_data(self.list_IDs)
        return X, y.flatten()

    def generate_data(self, temporary_metadata):
        """Generates data containing batch_size averaged time series.

        Args:
        -------
        list_IDs_temp: list
            list of label ids to load

        return: batch of averaged time series
        """
        X_data = np.zeros((0, self.n_channels, self.n_timepoints))
        y_data = []

        for i, metadata_file in temporary_metadata.iterrows():
            filename = os.path.join(self.BASE_PATH, 'processed_raw_' + metadata_file['cnt_file'] + '.zarr')
            
            data_signal = self.load_signal(filename)
            
            if (len(data_signal) == 0) and self.warnings:
                print(f"EMPTY SIGNAL, filename: {filename}")

            X = self.create_averaged_epoch(data_signal)

            X_data = np.concatenate((X_data, X), axis=0)
            y_data.append(metadata_file['age_months'])

        if self.shuffle:
            idx = np.arange(len(y_data))
            np.random.shuffle(idx)
            X_data = X_data[idx, :, :]
            y_data = [y_data[i] for i in idx]

            
        return np.swapaxes(X_data,1,2), np.array(y_data).reshape((-1,1))
    
    def create_averaged_epoch(self,
                              data_signal):
        """
        Function to create averages of self.n_average epochs.
        Will create one averaged epoch per found unique label from self.n_average random epochs.

        Args:
        --------
        data_signal: numpy array
            Data from one person as numpy array
        """
                                               
        # Create new data collection:
        X_data = np.zeros((0, self.n_channels, self.n_timepoints))
        num_epochs = len(data_signal)
                                               
        if num_epochs >= self.n_average:
            select = np.random.choice(num_epochs, self.n_average, replace=False)
            signal_averaged = np.mean(data_signal.oindex[select,:,:], axis=0)
        else:
            if self.warnings:
                print("Found only", num_epochs, " epochs and will take those!")            
            signal_averaged = np.mean(data_signal.oindex[:,:,:], axis=0)
                                                                                              
        X_data = np.concatenate([X_data, np.expand_dims(signal_averaged, axis=0)], axis=0)
                                    
        if self.gaussian_noise != 0.0:
            X_data += np.random.normal(0, self.gaussian_noise, X_data.shape)

        return X_data


    def load_signal(self,
                    filename):
        """Load EEG signal from one person.

        Args:
        -------
        filename: str
            filename...

        return: loaded array
        """
        return zarr.open(os.path.join(filename), mode='r')