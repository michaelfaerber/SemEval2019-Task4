#------------------------------------------------------------------------------------------
# Name:        Train doc2vec embeddings and an ML classifier based on them
# Purpose:     This module is used to train embeddings based on Doc2Vec. Two
#              separate embeddings are obtained for the title and the content.
#              These are then used to train an SVC model. Both embedding models
#              and the SVC model are committed to disk
#
# Execution:   python training.py
#                     [-h] [--path PATH] [--skipml] [--retrainpv]
#              path: Path to Semeval directory, skip: skip Machine leraning training,
#              retrain: retrain embeddings or load from previous run's pickle (if available).
#              If the pickle is not available, it will train embeddings.
#              (skipml=False by default, path: '/home/ashwath/Files/SemEval', retrainpv=False)
#
# Author:      Ashwath Sampath
#
# Created:     25-11-2018 (V1.0): Trains 2 dbow models which are committed to disk.
#                                 Trains an SVC model based on the embeddings
#                                 produced by these 2 models.
#                                 
# Revisions:   4-12-2018 (V1.1): Now pickling the paragraph vector(pv), taking new args
#                                from the command line for directory path, options to skip
#                                ML, retrain pv. y_train is a dataframe, not 
#                                a Series. 3 new pickles. Cleaned up code, paths.
#------------------------------------------------------------------------------------------

from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import pickle
import os
import argparse
import pandas as pd
from para2vec import ParagraphVectorModel, get_vector_label_mapping
import clean_shuffle

def build_pv_models(df, sem_eval_path):
    """ Function which builds the paragraph vector models (title and content models) based on the 
    data in the data frame df. Also pickles the paragraph vector instance.
    ARGUMENTS: df, Pandas Dataframe which has already been shuffled.
    RETURNS: ParagraphVectorModel object pv, which has 2 Doc2Vec models, 2 TaggedDocuments, and
             a dataframe as its members
    DETAILS: The 2 Doc2Vec models can be accessed by pv.model_content_dbow and pv.model_title_dbow.
             These are committed to disk when build_doc2vec_content_model and build_doc2vec_title_model are
              called (as Embeddings/doc2vec_dbow_model_content_idtags and Embeddings/doc2vec_dbow_model_title_idtags resp.)"""
    pv = ParagraphVectorModel(df, sem_eval_dir_path=sem_eval_path)
    # Remove df to save memory
    del df
    # Get docs of form [Word list, tag]: title and content tagged separately
    pv.get_tagged_docs()
    # Each of the models created in the foll. statemw
    pv.build_doc2vec_content_model()
    pv.build_doc2vec_title_model()
    pv_location = os.path.join(sem_eval_path, 'models', 'pv_object.pickle')
    with open(pv_location, 'wb') as pfile:
        pickle.dump(pv, pfile, pickle.HIGHEST_PROTOCOL)
    return pv

def train_ml_model(X_train, y_train):
    """ Trains a machine learning model based on a set of embeddings. Returns the model
    ARGUMENTS: X_train: embeddings, ndarray
               y_train: corresponding labels"""
    svc = SGDClassifier(alpha=0.0001, loss = 'hinge', max_iter=1000, n_jobs=-1)
    svc.fit(X_train, y_train)
    return svc

def main():
    """ Main function which reads the training file into a shuffled data frame, builds 2 ParagraphVectorModels,
    combines them, gets the resulting vector-label mappings, and trains an SVM (SVC) model on these mappings.
    This SVM model is persisted to disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",'-p', default="/home/ashwath/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    parser.add_argument("--skipml", '-s', action="store_true", default=False,
                        help="Use this argument to skip training the ML model")
    parser.add_argument("--retrainpv", '-r', action="store_true", default=False,
                        help="Use this argument to retrain the embeddings (loaded from previous run's pickle by default)")
    #parser.add_argument('--noskipml', dest='skipml', action='store_false')

    args = parser.parse_args()
    sem_eval_path = args.path
    filename = os.path.join(sem_eval_path, 'data', 'IntegratedFiles', 'buzzfeed_training_withid.tsv')
    df_location = os.path.join(sem_eval_path, 'data', 'Pickles', 'training_df.pickle')
    pv_location = os.path.join(sem_eval_path, 'models', 'pv_object.pickle')
    if args.retrainpv:
        df = clean_shuffle.read_prepare_df(filename, file_path=df_location)
        print("Training paragraph vectors...")
        pv = build_pv_models(df, sem_eval_path)
    else:
        try:
            # If a paragraph vector has already been pickled, load it in.
            with open(pv_location, 'rb') as pfile:
                print("Loading paragraph vector instance from pickle...")
                pv = pickle.load(pfile)
        except FileNotFoundError:
            # Doc2Vec training required
            df = clean_shuffle.read_prepare_df(filename, file_path=df_location)
            import sys
            sys.exit()
            print("Training paragraph vectors...")
            pv = build_pv_models(df, sem_eval_path)

    # Train machine learning model if args.skipml is False (default)
    if not args.skipml:
        # Get a composite embedding model: X_train has the vectors, y_train is a dataframe with id and
        # hyperpartisan indicator.
        print("Getting vector label mapping...")
        X_train, y_train_df = get_vector_label_mapping(pv, 'concat')
        # y_train_df is a dataframe, y_train_df.hyperpartisan has the labels.
        print("Training SVC...")
        svc = train_ml_model(X_train, y_train_df.hyperpartisan)
        # Serialize the model and save to disk
        svc_model_location = os.path.join(sem_eval_path, 'models', 'svc_embeddings.joblib')
        joblib.dump(svc, svc_model_location)      
    else:
        print("SVC model not trained")
    print("DONE!")    
    if not args.skipml:
        print("SVC Model saved to {}".format(svc_model_location))
    print("Paragraph vector object pickle is at: {}".format(pv_location))
    print("Dataframe is pickled at: {}".format(df_location))

if __name__ == '__main__':
    main()