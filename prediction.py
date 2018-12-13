#------------------------------------------------------------------------------------
# Name:        Prediction
# Purpose:     This module is used to predict the hyperpartisan values for a test or
#              validation set, and write the predictions to a file.
#
# Execution:   
#
# Author:      Ashwath Sampath
#
# Created:     12-12-2018 (V1.0): partly based on validation.py 
#------------------------------------------------------------------------------------

import pandas as pd
import os
import argparse
from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix, classification_report
import clean_shuffle
from para2vec import ParagraphVectorModel, get_vector_label_mapping, get_vector_tag_mapping
from datetime import datetime
import create_unified_tsv

import os
import getopt
import sys
from time import sleep

runOutputFileName = "prediction.txt"
sem_eval_path = '/home/peter-brinkmann'


def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir)


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

def test(test_file, outfile):
    """ Performs validation on the file supplied in the first argument.
    ARGUMENTS: test_file: the path to the test file, string
               out_file: path to output file
    RETURNS: None
    """
    test_df = clean_shuffle.read_prepare_test_df(test_file)
    # Load the model, and tag the docs (obviously, no training step, so set
    # init_models to False)
    pv = ParagraphVectorModel(test_df, init_models=False)
    # Remove the df to save memory
    del test_df
    # Tag the documents (title + content separately)
    pv.get_tagged_docs()
    pv.model_content_dbow = model_content_dbow
    pv.model_title_dbow = model_title_dbow
    # y_test_df is a DataFrame with id as the only column
    X_val, y_test_df = get_vector_tag_mapping(pv)
    # Get the predictions
    y_pred = predict_vals(svc, X_val)
    # Convert 0 and 1 back to true and false (as it was in the xml file)
    # ATTENTION: we don't need to convert it to 0 and 1 in the previous step any more.
    truefalsedict = {0: 'false', 1: 'true'}
    y_pred_df = pd.DataFrame(y_pred, columns=['predicted_hyperpartisan'])
    y_pred_df['predicted_hyperpartisan'] = y_pred_df['predicted_hyperpartisan'].map(truefalsedict, na_action=None)
    # The order of ids will be the same, also add leading zeros (to make it like the input dataset)
    y_pred_df['id'] = y_test_df['id'].astype(str).str.zfill(7)
    # Reorder the columns
    y_pred_df = y_pred_df[['id', 'predicted_hyperpartisan']]
    # Write to file
    y_pred_df.to_csv(outfile, sep=' ', index=False, header=False)

########## MAIN ##########


def main(inputDataset, outputDir):
    """Main method of this module."""
    # Load the models in the global scope
    loadmodels_global()
    outfile = outputDir + "/" + runOutputFileName
    
    for file in os.listdir(inputDataset):
        if file.endswith(".xml"):
            xml_file = inputDataset + "/" + file
            if 'test' in xml_file:
                if 'article' in xml_file:
                    intermediate_tsv = '{}/data/crowdsourced_test_withid'.format(sem_eval_path)
                else:
                    intermediate_tsv = '{}/data/buzzfeed_test_withid'.format(sem_eval_path)
            if 'validation' in xml_file:
                if 'article' in xml_file:
                    intermediate_tsv = '{}/data/crowdsourced_validation_withid'.format(sem_eval_path)
                else:
                    intermediate_tsv = '{}/data/buzzfeed_validation_withid'.format(sem_eval_path)
            if 'train' in xml_file:
                if 'article' in xml_file:
                    intermediate_tsv = '{}/data/crowdsourced_validation_withid'.format(sem_eval_path)
                else:
                    intermediate_tsv = '{}/data/buzzfeed_validation_withid'.format(sem_eval_path)                    
            create_unified_tsv.write_to_tsv(intermediate_tsv, xml_file)
            print("Written to TSV intermediate file")
            sleep(2)

    # Do the testing/validation: intermediate_tsv is the input file, outfile is the output file for the predictions.
    test(intermediate_tsv, outfile)

    print("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())
