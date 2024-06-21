import os
import numpy as np
import pandas as pd
import scipy.io
import pickle
import argparse
import logging
os.chdir(os.path.dirname(os.getcwd()))

# ts_class packages, imported from the downloaded Github repository
from activity_classifier.main import run_model
from activity_classifier.config import TSF_MODEL, RISE_MODEL, OBS, PREDICTION, OUTPUT_PATH
from activity_classifier.prepare_data import prepare_data
from activity_classifier.retrain_models import retrain_tsf, retrain_rise
from math import floor
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score

def run_many_classifiers_shuffle(experiments,dFoverFs, duration, sampling_rate, training_file_path, output_folder_path,start_loop, end_loop):
    for loop in range(start_loop,end_loop):
        
        shuffle_training_data(training_file_path)
        
        this_class_SA_fractions,this_class_predictions = run_new_classifier(experiments, dFoverFs, duration, sampling_rate, training_file_path)
        results_path = os.path.join(output_folder_path,'Classifiers_Shuffled/SA_classifier_predictions_'+str(loop)+'.pickle')
        with open(results_path,'wb') as handle:
            pickle.dump([this_class_SA_fractions, this_class_predictions], handle, protocol=pickle.HIGHEST_PROTOCOL)
        #print('Done run_many_classifiers loop '+str(loop))    
        
def run_many_classifiers(experiments, dFoverFs,duration, sampling_rate, training_file_path, output_folder_path,start_loop, end_loop):
    for loop in range(start_loop,end_loop):
        this_class_SA_fractions,this_class_predictions = run_new_classifier(experiments, dFoverFs,duration, sampling_rate, training_file_path)
        results_path = os.path.join(output_folder_path,'Classifiers/SA_classifier_predictions_'+str(loop)+'.pickle')
        with open(results_path,'wb') as handle:
            pickle.dump([this_class_SA_fractions, this_class_predictions], handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print('Done run_many_classifiers loop '+str(loop))  
            
def run_new_classifier(experiments, dFoverFs,duration, sampling_rate, training_file_path):
    #retrain model
    retrain_ts_class_model(training_file_path, duration, sampling_rate)
    
    #make predictions
    this_class_SA_fractions = {}
    this_class_predictions = {}
    for experiment in experiments:
        print(experiment)
        this_class_SA_fractions[experiment]=[]
        this_class_predictions[experiment] =[]
        snip_count=0
        
        for snippet in get_consecutive_snippets(dFoverFs[experiment],duration,sampling_rate,1200):
            snip_count=snip_count+1    
            data = run_batch_ts_class_prediction(pd.DataFrame(data=snippet),duration, sampling_rate)
        
            this_class_SA_fractions[experiment].append(get_SA_fraction(data,'RISE'))
            this_class_predictions[experiment].append(get_SA_predictions(data,'RISE'))
            
    this_class_SA_fractions['snippet']=list(range(0,snip_count))
    # print('Done run_new_classifier')
    return(this_class_SA_fractions,this_class_predictions)

def run_batch_ts_class_prediction(raw_data,duration,sampling_rate):
    interpolate = False #@param ["True", "False"] {type:"raw"}
    run_total_frames = floor(sampling_rate * duration)
    
    trace_data = raw_data.iloc[:, 0:run_total_frames]
    data = deepcopy(prepare_data(trace_data, duration, sampling_rate))
    data = run_model(data, RISE_MODEL, 'RISE')
    # print('Done run_batch_ts_class_prediction')
    return(data)


def retrain_ts_class_model(training_file_path, duration, sampling_rate):
    # print('Starting retrain_ts_class_model')
    train_total_frames = floor(duration * sampling_rate)
    # print("1. Reading csv file...")
    data = pd.read_csv(training_file_path, header=0)
    # print("2. Normalising data...")
    trace_data = deepcopy(data.iloc[:, 0:train_total_frames])
    data = pd.concat([prepare_data(trace_data, duration, sampling_rate), data['status']], axis=1)
    # print("3. Retraining Random Interval Spectral Ensemble...")
    retrain_rise(data)
    #retrain_rise_without_cv(data)
    # print('Done retrain_ts_class_model')


def get_consecutive_snippets(dFoverF, snippet_duration, sampling_rate, final_frame):
    all_snippets =[]
    frames_per_snippet = floor(sampling_rate * snippet_duration)
    end_frame = frames_per_snippet
    start_frame =0
    while end_frame <= final_frame:
        this_snippet = get_dFoverF_snippet(dFoverF,start_frame, end_frame)
        start_frame = end_frame
        end_frame = end_frame + frames_per_snippet
        all_snippets.append(this_snippet)
    return (np.array(all_snippets))


def get_dFoverF_snippet(dFoverF,start_frame,end_frame): # where end_frame is not included!
    snippet = dFoverF[:,start_frame:end_frame]
    return snippet


def get_SA_fraction(data,model):
    if 'active' in data['RISE_prediction'].unique(): # if any active cells exist
        SA_fraction = data['RISE_prediction'].value_counts()['active']/data['RISE_prediction'].value_counts().sum()
    else:
        SA_fraction = 1-(data['RISE_prediction'].value_counts()['inactive']/data['RISE_prediction'].value_counts().sum()) #this should be zero but rather than setting it to zero do this so an error occurs if it breaks (sorry!)
    return(SA_fraction)


def get_SA_predictions(data,model):
    predictions = data['RISE_prediction']
    return(predictions)


def shuffle_training_data(training_file_path):
    training_data = pd.read_csv(training_file_path, header=0)
    training_labels = training_data['status'].values
    np.random.shuffle(training_labels)
    shuffled_data = training_data.iloc[:,:-1]
    shuffled_data.insert(shuffled_data.shape[1],'status',training_labels)
    shuffled_data.to_csv("data/training_data_shuffled.csv", index=False)
    
    
def get_predictions(output_folder_path,experiments,shuffle=False):
    
    if shuffle:
        folder='Classifiers_Shuffled'
    else:
        folder='Classifiers'
            
    results_files = [f for f in os.listdir(os.path.join(output_folder_path,folder)) if os.path.isfile(os.path.join(os.path.join(output_folder_path,folder),f))]
    SA_fractions = []
    predictions =[]
    for instance in results_files:
        with open(os.path.join(output_folder_path, folder, instance), 'rb') as handle:
            data = pickle.load(handle)
            this_class_SA_fractions,this_class_predictions = data
            SA_fractions.append(this_class_SA_fractions)
            predictions.append(this_class_predictions)

    nloops = np.array(results_files).shape[0]
    
    pred_array ={}
    for experiment in experiments:
        ncells = predictions[0][experiment][0].shape[0]
        nsnippets= np.array(predictions[0][experiment]).shape[0]
        pred_array[experiment]= np.zeros([nloops*nsnippets,ncells])
    
        for l in range(0,nloops):
            for s in range(0,nsnippets):
                this_pred = []
                for c in predictions[l][experiment][s]:
                    if c=='active':
                        this_pred.append(1)
                    elif c=='inactive':
                        this_pred.append(0)    
                pred_array[experiment][nloops*s+l,:]=this_pred  
                
    return(SA_fractions,predictions,pred_array)


def get_performance_measures(output_folder_path,pred_array, experiments_with_labels, lidocaine_labels, shuffle=False):
    Accuracies = {}
    Precisions = {}
    Recalls = {}
    Specificities={}
    for experiment in experiments_with_labels:
        this_accuracy = []
        this_precision = []
        this_recall = []
        this_specificity =[]
        y_true = lidocaine_labels[experiment]
        nloops = pred_array[experiment].shape[0]
        for l in range(0,nloops):
            y_pred = pred_array[experiment][l]
            this_accuracy.append(accuracy_score(y_true,y_pred))
            this_precision.append(precision_score(y_true,y_pred))
            this_recall.append(recall_score(y_true,y_pred))
            this_specificity.append(recall_score(y_true, y_pred, pos_label=0))
        Accuracies[experiment]=this_accuracy
        Precisions[experiment]=this_precision
        Recalls[experiment]=this_recall
        Specificities[experiment]=this_specificity
    
    shuffle_flag = '_shuffle' if shuffle else ''
    pd.DataFrame(data=Accuracies).to_csv(os.path.join(output_folder_path,'Accuracies_'+str(nloops)+'loops'+shuffle_flag+'.csv'))
    pd.DataFrame(data=Precisions).to_csv(os.path.join(output_folder_path,'Precisions_'+str(nloops)+'loops'+shuffle_flag+'.csv'))
    pd.DataFrame(data=Recalls).to_csv(os.path.join(output_folder_path,'Recalls_'+str(nloops)+'loops'+shuffle_flag+'.csv'))
    pd.DataFrame(data=Specificities).to_csv(os.path.join(output_folder_path,'Specificities_'+str(nloops)+'loops'+shuffle_flag+'.csv'))
    
    return(Accuracies,Precisions,Recalls,Specificities)