"""
Helper functions to work with ePODIUM EEG data

"""
import numpy as np


def select_bad_channels(data_raw, time = 100, threshold = 5, include_for_mean = 0.8):
    """
    Function to find suspect channels --> still might need manual inspection!
    
    Args:
    --------
    data_raw: mne object
        
    time: int
        Time window to look for ouliers (time in seconds). Default = 100.
    threshold: float/int
        Relative threshold. Anything channel with variance > threshold*mean OR < threshold*mean
        will be considered suspect. Default = 5.
    include_for_mean: float
        Fraction of variances to calculate mean. This is to ignore the highest and lowest
        ones, which coul dbe far outliers.
    
    """
    sfreq = data_raw.info['sfreq']
    no_channels = len(data_raw.ch_names) -1  # Subtract stimuli channel
    data, times = data_raw[:no_channels, int(sfreq * 10):int(sfreq * (time+10))]
    variances = []
    for i in range(data.shape[0]):
        variances.append(data[i,:].var())
    var_arr = np.array(variances)
    exclude = int((1-include_for_mean)*no_channels/2)
    mean_low = np.mean(np.sort(var_arr)[exclude:(no_channels-exclude)])
    
    suspects = np.where((var_arr > threshold* mean_low) & (var_arr < threshold/mean_low))[0]
    suspects_names = [data_raw.ch_names[x] for x in list(suspects)]
    selected_suspects = [data_raw.ch_names.index(x) for x in suspects_names if not x in ['HEOG', 'VEOG']]
    selected_suspects_names = [x for x in suspects_names if not x in ['HEOG', 'VEOG']]
    print("Suspicious channel(s): ", selected_suspects_names)
    
    return selected_suspects, selected_suspects_names



def select_bad_epochs(epochs, stimuli, threshold = 5, max_bad_fraction = 0.2):
    """
    Function to find suspect epochs and channels --> still might need manual inspection!
    
    Args:
    --------
    epochs: epochs object (mne)
    
    stimuli: int/str
        Stimuli to pick epochs for.         
    threshold: float/int
        Relative threshold. Anything channel with variance > threshold*mean OR < threshold*mean
        will be considered suspect. Default = 5.   
    max_bad_fraction: float
        Maximum fraction of bad epochs. If number is higher for one channel, call it a 'bad' channel
    """
    bad_epochs = set()
    bad_channels = []
    
    from collections import Counter
    
    signals = epochs[str(stimuli)].get_data()
    max_bad_epochs = max_bad_fraction*signals.shape[0]
    
    # Find outliers in episode STD and max-min difference:
    signals_std = np.std(signals, axis=2)
    signals_minmax = np.amax(signals, axis=2) - np.amin(signals, axis=2)
    
    outliers_high = np.where((signals_std > threshold*np.mean(signals_std)) | (signals_minmax > threshold*np.mean(signals_minmax)))
    outliers_low = np.where((signals_std < 1/threshold*np.mean(signals_std)) | (signals_minmax < 1/threshold*np.mean(signals_minmax)))
    outliers = (np.concatenate((outliers_high[0], outliers_low[0])), np.concatenate((outliers_high[1], outliers_low[1])) ) 
    
    if len(outliers[0]) > 0:
        print("Found", len(set(outliers[0])), "bad epochs in a total of", len(set(outliers[1])), " channels.")
        occurences = [(Counter(outliers[1])[x], x) for x in list(Counter(outliers[1]))]
        for occur, channel in occurences:
            if occur > max_bad_epochs:
                print("Found bad channel (more than", max_bad_epochs, " bad epochs): Channel no: ", channel )
                bad_channels.append(channel)
            else:
                # only add bad epochs for non-bad channels
                bad_epochs = bad_epochs|set(outliers[0][outliers[1] == channel])
                
        print("Marked", len(bad_epochs), "bad epochs in a total of", signals.shape[0], " epochs.")
        
#        # Remove bad data:
#        signals = np.delete(signals, bad_channels, 1)
#        signals = np.delete(signals, list(bad_epochs), 0)
        
    else:
        print("No outliers found with given threshold.")
    
    return [epochs.ch_names[x] for x in bad_channels], list(bad_epochs)

def select_bad_epochs_list(epochs, stimuli, threshold = 5, max_bad_fraction = 0.2):
    """
    Function to find suspect epochs and channels --> still might need manual inspection!
    
    Args:
    --------
    epochs: epochs object (mne)
    
    stimuli: list of int/str
        Stimuli to pick epochs for.         
    threshold: float/int
        Relative threshold. Anything channel with variance > threshold*mean OR < threshold*mean
        will be considered suspect. Default = 5.   
    max_bad_fraction: float
        Maximum fraction of bad epochs. If number is higher for one channel, call it a 'bad' channel
    """
    
    from collections import Counter
 
    bad_epochs = set()
    bad_channels = []
     
    for stimulus in stimuli:
        signals = epochs[str(stimulus)].get_data()
        max_bad_epochs = max_bad_fraction*signals.shape[0]

        # Find outliers in episode STD and max-min difference:
        signals_std = np.std(signals, axis=2)
        signals_minmax = np.amax(signals, axis=2) - np.amin(signals, axis=2)

        outliers_high = np.where((signals_std > threshold*np.mean(signals_std)) | (signals_minmax > threshold*np.mean(signals_minmax)))
        outliers_low = np.where((signals_std < 1/threshold*np.mean(signals_std)) | (signals_minmax < 1/threshold*np.mean(signals_minmax)))
        outliers = (np.concatenate((outliers_high[0], outliers_low[0])), np.concatenate((outliers_high[1], outliers_low[1])) ) 

        if len(outliers[0]) > 0:
            print("Found", len(set(outliers[0])), "bad epochs in a total of", len(set(outliers[1])), " channels.")
            occurences = [(Counter(outliers[1])[x], x) for x in list(Counter(outliers[1]))]
            for occur, channel in occurences:
                if occur > max_bad_epochs:
                    print("Found bad channel (more than", max_bad_epochs, " bad epochs): Channel no: ", channel )
                    bad_channels.append(channel)
                else:
                    # only add bad epochs for non-bad channels
                    bad_epochs = bad_epochs|set(outliers[0][outliers[1] == channel])

            print("Marked", len(bad_epochs), "bad epochs in a total of", signals.shape[0], " epochs.")

    #        # Remove bad data:
    #        signals = np.delete(signals, bad_channels, 1)
    #        signals = np.delete(signals, list(bad_epochs), 0)

        else:
            print("No outliers found with given threshold.")
    
    return [epochs.ch_names[x] for x in bad_channels], list(bad_epochs)