#------------------------------------------------------------------------------------
# Name:        Validation
# Purpose:     This module contains is use to perform validation using the provided
#              validation set, and to calculate a number of metrics. Further, the
#              hand-prepared training file with 645 records is used as a second
#              validation file, mimicking the 2 test files.
#
# Execution:   Not executable
#
# Author:      Ashwath Sampath
#
# Created:     25-11-2018 (V1.0): Validation performed on 2 validation sets, a number
#                                 of metrics are written to logs/validation_log.log
#                                 
# Revisions:   04-12-2018 (V1.1): Cleaned up paths, read/write dfs from pickle,
#                                 Combine dataframes and write predictions to file
#                                 with id, add globals.
#------------------------------------------------------------------------------------

import pandas as pd
import os
import argparse
from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix, classification_report
import clean_shuffle
from para2vec import ParagraphVectorModel, get_vector_label_mapping
from datetime import datetime

log_name = '/home/ashwath/Files/SemEval/logs/validation_log_{}.log'.format(
    datetime.now().strftime("%Y-%m-%d_%H%M%S"))
results_log = open(log_name, 'a')

def loadmodels_global():
    """ Load the models in the global scope. sem_eval_path is global. """
    global model_content_dbow
    model_content_dbow = Doc2Vec.load(os.path.join(sem_eval_path, 'embeddings', 'doc2vec_dbow_model_content_idtags'))
    global model_title_dbow
    model_title_dbow = Doc2Vec.load(os.path.join(sem_eval_path, 'embeddings', 'doc2vec_dbow_model_title_idtags'))
    global svc
    svc = joblib.load(os.path.join(sem_eval_path, 'models', 'svc_embeddings.joblib'))

def predict_vals(model, X_val):
    """ Predicts the labels for the validation set using the given model
    ARGUMENTS: model: an sklearn model
               X_val: the validation matrix for which labels have to be predicted
    RETURNS: y_pred: predicted labels Pandas series"""
    return pd.Series(model.predict(X_val))

def calculate_metrics(y_test_df, y_pred_df, ml_model, val_filetype):
    """ Calculates a number of metrics using the model, the predicted y and the true y.
    ARGUMENTS: y_test_df: test (validation) set labels and ids, Pandas DataFrame
               y_pred_df: predicted labels and ids, Pandas DataFrame
               ml_model: sklearn model (hyperparams printed in log file)
               val_filetype: string 'Buzzfeed Validation File' or
               'Crowdsourced File used as a validation file'
    RETURNS: None
               """
    y_pred = y_pred_df.hyperpartisan
    y_test = y_test_df.hyperpartisan
    results_log.write("{}: \n".format(val_filetype))
    results_log.write("ML Model for classification: {}\n".format(ml_model))
    results_log.write("Predicted value counts per class (predictions):\n{}\n ".format(y_pred.value_counts()))
    results_log.write("Predicted value counts per class (val set):\n{}\n ".format(y_test.value_counts()))
    f1 = f1_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    accuracy = accuracy_score(y_test, y_pred)
    results_log.write("F1={}, Precision={}, Recall={}, Accuracy={}".format(f1, precision, recall, accuracy))
    results_log.write(classification_report(y_test, y_pred, target_names=['fair', 'biased'] ))
    results_log.write("Confusion matrix: \n{}\n".format(confusion_matrix(y_test, y_pred)))
    results_log.write('****************************************************************************************************************\n')

def write_to_tsv(y_pred_df, y_val_df, outfile):
    """ Combine the contents of the predicted y and original y dataframes, and write the results to outfile
    ARGUMENTS: y_test_df: test (validation) set labels and ids, Pandas DataFrame
               y_pred_df: predicted labels and ids, Pandas DataFrame
               outfile: output file to write to (whole path), string
    RETURNS: None
    """
    # Convert 0 and 1 back to true and false (as it was in the xml file)
    truefalsedict = {0: 'false', 1: 'true'}
    y_pred_df['hyperpartisan'] = y_pred_df['hyperpartisan'].map(truefalsedict, na_action=None)
    y_val_df['hyperpartisan'] = y_val_df['hyperpartisan'].map(truefalsedict, na_action=None)
    y_pred_df = y_pred_df.rename(columns={'hyperpartisan': 'predicted_hyperpartisan'})
    y_val_df = y_val_df.rename(columns={'hyperpartisan': 'actual_hyperpartisan'})
    df = pd.merge(y_val_df, y_pred_df, how='inner', left_on='id', right_on='id')
    # Reorder columns
    df = df[['id', 'actual_hyperpartisan', 'predicted_hyperpartisan']]
    df.to_csv(outfile, sep='\t', index=False)

def validate(val_file, val_filetype, df_location, outfile):
    """ Performs validation on the file supplied in the first argument.
    ARGUMENTS: val_file: the path to the validation file, string
               val_filetype: string 'Buzzfeed Validation File' or
               'Crowdsourced File used as a validation file'
               df_location: location to load/save validation df from
               out_file: path to output file
    RETURNS: None
    """
    val_df = clean_shuffle.read_prepare_df(val_file, file_path=df_location)
    # Load the model, and tag the docs (obviously, no training step, so set
    # init_models to False)
    pv = ParagraphVectorModel(val_df, init_models=False)
    # Tag the documents (title + content separately)
    pv.get_tagged_docs()
    pv.model_content_dbow = model_content_dbow
    pv.model_title_dbow = model_title_dbow
    # y_val_df is a DataFrame with id and hyperpartisan
    X_val, y_val_df = get_vector_label_mapping(pv)
    # Get the predictions
    y_pred = predict_vals(svc, X_val)
    y_pred_df = pd.DataFrame(y_pred, columns=['hyperpartisan'])
    # The order of ids will be the same
    y_pred_df['id'] = y_val_df.id
    calculate_metrics(y_val_df, y_pred_df, svc, val_filetype)
    write_to_tsv(y_pred_df, y_val_df, outfile)

def main():
    """ Main function which performs validation on 2 validation files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputDataset",'-c', default="/home/ashwath/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    parser.add_argument("--outputDir",'-o', default="/home/ashwath/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")

    args = parser.parse_args()
    global sem_eval_path
    sem_eval_path = '/home/peter-brinkmann'
    val_file = os.path.join(sem_eval_path, 'data', 'IntegratedFiles', 'buzzfeed_validation_withid.tsv')
    crowdsourced_file = os.path.join(sem_eval_path, 'data', 'IntegratedFiles', 'crowdsourced_train_withid.tsv')
    val_df = os.path.join(sem_eval_path, 'data', 'Pickles', 'validation_df.pickle')
    crowdsourced_df = os.path.join(sem_eval_path, 'data', 'Pickles', 'crowdsourced_df.pickle')
    outfile_buzzfeedval =  os.path.join(sem_eval_path, 'predictions', 'buzzfeedval_predictions.tsv')
    outfile_crowdsourced = os.path.join(sem_eval_path, 'predictions', 'crowdsourced_predictions.tsv')
    # Load the models in the global scope
    loadmodels_global()
    validate(val_file, 'Buzzfeed Validation File', val_df, outfile=outfile_buzzfeedval)
    validate(crowdsourced_file, 'Crowdsourced File used as a validation file', crowdsourced_df, outfile=outfile_crowdsourced)
    print("DONE! Metrics and results stored in {}".format(log_name))
    print("View Buzzfeed validation predictions at: {}".format(outfile_buzzfeedval))
    print("View Crowdsourced validation predictions at: {}".format(outfile_crowdsourced))
    results_log.close()

if __name__ == '__main__':
    main()