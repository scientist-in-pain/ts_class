# TimeSeries Classification for Spontaneous Neuron Activity

This repo contains code and pickle files for two binary classifiers of time series data. The aim of these models is to predict whether or not a neuron has spontaneous activity.

Two models are included in this repo, from the SKTime project: TimeSeries Forest Classifier and Random Spectral Interval Ensemble. For more detail on these models, see  https://www.sktime.org/en/latest/ .

# How to use this repo

## Cloning the repo
The repo should be cloned to your local machine, see here: https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository


## Install requirements
It is assumed that python is already installed on your local machine. A virtual environment is recommended to hold all libraries required to run the project in a clean and safe way. To do this:
1. Open command line
2. python -m venv /path/to/new/virtual/environment
3. /path/to/new/virtual/environment/Scripts/activate
4. pip install -r path/to/requirements.txt


## Running models
Both models are set up to be run from main.py, with the output of both contained in an output .csv file in the /data directory.

The input data should be a .csv file of the format:
x_axis = observations_over_time (up to 1306 observations)
y_axis = neurons
No header rows or columns should be included.

To run the script, activate the virtual environment if using, then in the command line move to the ts_class folder and enter
 python -m activity_classifier.main path/to/data.csv
 
 The output file should appear in the ts_class/data directory
 
 
 ## Retraining models
 Models can be retrained using the retrain_models script.
 
 Training data should be in the same format as above, but also include a labelling column called 'status' which contains the activity label for the neuron on that row (e.g. 'active', 'inactive')
 
 To run the script, activate the virtual environment if using, then in the command line move to the ts_class folder and enter
 python -m activity_classifier.retrain_models path/to/training/data.csv
 
 The retrained models will be saved in the /models directory and will replace any existing models.
 
 
 



