# Handover document

The initial idea of the study was to use the EEG data set for predicting dyslexia risk. As earlier studies showed that it was not feasible with the current data set, we have shifted the focus of the project to predicting the developmental age in infants using the same data set. The notebooks used for this project with this shifted focus can be found in this repository. The results can be found in Bjorn Bruns' master's thesis

> Bruns, B. M. A. (2021). _Predicting developmental age in young children by applying deep learning approaches to EEG data_ (Master's thesis).

## Next steps (suggestions)

There are several improvements or next steps possible for this project. However, the most interesting improvement might be to use a new data (latest techniques) set specifically collected for the question at hand. It might be possible to go back to the initial question: is it possible to predict dyslexia (risk) using EEG data?

Other suggestions are (non-exclusive list):

- Retrain (similar) models to predict dyslexia using a new data set collected with the lastest EEG techniques
- Attempt to predict the developmental age with a new data set collected with the latest EEG techniques
- Retrain the models for other task (e.g. finding other biomarkers)
- Make the models more explainable (e.g. visualizing weights of multiple models, relate the findings to physiological events in the human brain)
- The current models did not leverage information from the oddball paradigm (would've decreased the data set significantly)
- Use more EEG channels as input for the model, for this project we only used a subset (the minimum number that was available for all subjects)
- General model improvements, data augmentation

## Repositories

The [ePODIUM project on GitHub](https://github.com/epodium) currently contains 4 repositories. This handover document was written after the age prediction project was finished. A brief description of the 4 repositories:

- [EEG_explorer](https://github.com/epodium/EEG_explorer) - focuses on exploring EEG data using Python (the MNE package specifically). Great for getting acquainted with working in Python with this type of data.
- [EEG_dyslexia_prediction](https://github.com/epodium/EEG_dyslexia_prediction) - actively worked on until the beginning of 2020, goal was to use the EEG data set and predict dyslexia risk.
- [EEG_age_prediction](https://github.com/epodium/EEG_age_prediction) - focuses on predicting developmental age instead of dyslexia and contains all the code used for writing the master's thesis. Some of the code is taken from the two repositories above and modified for the new project's goal
- [time_series_generator](https://github.com/epodium/time_series_generator) - Side project to generate synthetic time series data by the eScience Center
