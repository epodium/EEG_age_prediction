# Import libraries
from tensorflow.keras.utils import Sequence
import numpy as np
import zarr
import os

class DataGeneratorReduced(Sequence):
    """Generates data for loading (preprocessed) EEG timeseries data.
    Create batches for training or prediction from given folders and filenames.

    """
    def __init__(self,
                 list_IDs,
                 BASE_PATH,
                 metadata,
                 gaussian_noise=0.0,
                 n_average = 3,
                 batch_size=10,
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

        # Store all data in here
        self.X_data_all = []
        self.y_data_all = []
        self.load_all_data()


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

        # Generate data
        X, y = self.generate_data(indexes)

        return X, y

    def load_all_data(self):
        """ Loads all data into memory. """
        for i, metadata_file in self.metadata_temp.iterrows():
            filename = os.path.join(self.BASE_PATH, metadata_file['cnt_file'] + '.zarr')
            
            X_data = np.zeros((0, self.n_channels, self.n_timepoints))

            data_signal = self.load_signal(filename)

            if (len(data_signal) == 0) and self.warnings:
                print(f"EMPTY SIGNAL, filename: {filename}")

            X_data = np.concatenate((X_data, data_signal), axis=0)

            self.X_data_all.append(X_data)
            self.y_data_all.append(metadata_file['age_months'])
        
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

    def generate_data(self, indexes):
        """Generates data containing batch_size averaged time series.

        Args:
        -------
        list_IDs_temp: list
            list of label ids to load

        return: batch of averaged time series
        """
        X_data = np.zeros((0, self.n_channels, self.n_timepoints))
        y_data = []

        for index in indexes:
            X = self.create_averaged_epoch(self.X_data_all[index])

            X_data = np.concatenate((X_data, X), axis=0)
            y_data.append(self.y_data_all[index])
            
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
            signal_averaged = np.mean(data_signal[select,:,:], axis=0)

        else:
            if self.warnings:
                print("Found only", num_epochs, " epochs and will take those!")          
            signal_averaged = np.mean(data_signal[:,:,:], axis=0)
                                                                                              
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