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
# Revisions:   
#------------------------------------------------------------------------------------

import pandas as pd
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

# Load the models in the global scope
model_content_dbow = Doc2Vec.load('/home/ashwath/Files/SemEval/embeddings/doc2vec_dbow_model_content')
model_title_dbow = Doc2Vec.load('/home/ashwath/Files/SemEval/embeddings/doc2vec_dbow_model_title')
svc = joblib.load('/home/ashwath/Files/SemEval/models/svc_embeddings.joblib')

def predict_vals(model, X_val):
    """ Predicts the labels for the validation set using the given model
    ARGUMENTS: model: an sklearn model
               X_val: the validation matrix for which labels have to be predicted
    RETURNS: y_pred: predicted labels Pandas series"""
    return pd.Series(model.predict(X_val))

def calculate_metrics(y_test, y_pred, ml_model, val_filetype):
    """ Calculates a number of metrics using the model, the predicted y and the true y.
    ARGUMENTS: y_test: test (validation) set labels, Pandas Series
               y_pred: predicted labels, Pandas Series
               ml_model: sklearn model (hyperparams printed in log file)
               val_filetype: string 'Buzzfeed Validation File' or
               'Crowdsourced File used as a validation file'
    RETURNS: None
               """

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

def validate(val_file, val_filetype):
    """ Performs validation on the file supplied in the first argument.
    ARGUMENTS: val_file: the path to the validation file, string
               val_filetype: string 'Buzzfeed Validation File' or
               'Crowdsourced File used as a validation file'
    RETURNS: None
    """
    val_df = clean_shuffle.read_prepare_df(val_file)
    # Load the model, and tag the docs (obviously, no training step, so set
    # init_models to False)
    pv = ParagraphVectorModel(val_df, init_models=False)
    # Tag the documents (title + content separately)
    pv.get_tagged_docs()
    pv.model_content_dbow = model_content_dbow
    pv.model_title_dbow = model_title_dbow
    X_val, y_val = get_vector_label_mapping(pv)
    # Get the predictions
    y_pred = predict_vals(svc, X_val)
    calculate_metrics(y_val, y_pred, svc, val_filetype)

def main():
    """ Main function which performs validation on 2 validation files."""
    val_file = '/home/ashwath/Files/SemEval/data/IntegratedFiles/buzzfeed_validation.tsv'
    crowdsourced_file = '/home/ashwath/Files/SemEval/data/IntegratedFiles/crowdsourced_train.tsv'
    validate(val_file, 'Buzzfeed Validation File')
    validate(crowdsourced_file, 'Crowdsourced File used as a validation file')
    print("DONE! Results in {}".format(log_name))
    results_log.close()

if __name__ == '__main__':
    main()