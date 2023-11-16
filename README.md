# TimeSeries Classification for Spontaneous Neuron Activity

This repo contains code, pickle files and a Google Colab notebook for two binary classifiers of time series data. The aim of these models is to predict whether or not a neuron has spontaneous activity.

Two models are included in this repo, from the SKTime project: TimeSeries Forest Classifier and Random Spectral Interval Ensemble. For more detail on these models, see  https://www.sktime.org/en/latest/ .

# How to use this repo

## Using the Colab notebook
The easiest way to run and train these models is via the Colab notebook. To use this, navigate to notebooks/SpontaneousActivity.ipynb. Once you've clicked into the file, there is a box at the top saying 'Open in Colab'. Click this, wait for the new page to load and follow the instructions in the notebook.

## Cloning the repo
If needed, the repo can be cloned to your local machine, see here: https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository


## Install requirements
If running locally, it is assumed that python is already installed on your local machine. A virtual environment is recommended to hold all libraries required to run the project in a clean and safe way. To do this:
1. Open command line
2. python -m venv /path/to/new/virtual/environment    #We recommned creating new virtual environment in the envs folder C:/Users/Your_name/anaconda3/envs/SA
4. /path/to/new/virtual/environment/Scripts/activate   #If using envs folder C:/Users/Your_name/anaconda3/envs/SA/Scripts/activate
5. cd path/to/ts_class/   #change working directory to location of ts_class folder
7. pip install -r requirements.txt


## Data
Data should be prepared in a csv file, in the following format:

0 | 1 | 2 | 3 | ... | status
--|---|---|---|-----|-------
1.2|1.3|1.4|2.0| ... | inactive
2.2|2.3|2.4|3.0| ... | active

The 'status' column is required for model training, but not for running the classifier.
x_axis = observations_over_time
y_axis = neurons

Example train and test datasets are provided in the 'data' folder. The current models are trained on 151 second recordings. If greater or less durations are required, model retraining is needed. This can be done using our data (up to 301 seconds), or the users own data can be used, formatted as above.

## Running models
Both models are set up to be run from main.py, with the output of both contained in an output .csv file in the /data directory.

To run the script, activate the virtual environment if using (use conda or source depending on how environment was created)
```commandline
conda/source activate 'name of virtual environment'
```
In the command line move to the ts_class folder 
```commandline
cd path/to/ts_class/
```
enter
 ```commandline
 python -m activity_classifier.main path/to/data.csv <recording duration> <recording sampling rate>
 ```
 Note, duration and sampling rate should be the same as those used in model training. If different duration or sampling rate are required, retrain first.
 The output file should appear in the ts_class/data directory. 

To run on our test data, use:
```commandline
python -m activity_classifier.main data/test_data.csv 151 3.65
```
 
 
 ## Retraining models
 Models can be retrained using the retrain_models script.
 
 Training data should be in the same format as above, but also include a labelling column called 'status' which contains the activity label for the neuron on that row (e.g. 'active', 'inactive')
 
 To run the script, activate the virtual environment if using, then in the command line move to the ts_class folder and enter
 ```commandline
 python -m activity_classifier.retrain_models path/to/training/data.csv <recording duration> <recording sampling rate>
 ```
 
The retrained models will be saved in the /models directory and will replace any existing models.

To train models using our sample data, use:
```commandline
python -m activity_classifier.retrain_models data/training_data.csv 151 3.65
```
 
 
 



