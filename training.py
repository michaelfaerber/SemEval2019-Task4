#------------------------------------------------------------------------------------
# Name:        Train doc2vec embeddings and an ML classifier based on them
# Purpose:     This module is used to train embeddings based on Doc2Vec. Two
#              separate embeddings are obtained for the title and the content.
#              These are then used to train an SVC model. Both embedding models
#              and the SVC model are committed to disk
#
# Execution:   python training.py
#
# Author:      Ashwath Sampath
#
# Created:     25-11-2018 (V1.0): Trains 2 dbow models which are committed to disk.
#                                 Trains an SVC model based on the embeddings
#                                 produced by these 2 models.
#                                 
# Revisions:   
#------------------------------------------------------------------------------------

import clean_shuffle
from para2vec import ParagraphVectorModel, get_vector_label_mapping
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import argparse

def build_pv_models(df, sem_eval_dir_path):
    """ Function which builds the paragraph vector models (title and content models) based on the 
    data in the data frame df. 
    ARGUMENTS: df, Pandas Dataframe which has already been shuffled.
    RETURNS: ParagraphVectorModel object pv, which has 2 Doc2Vec models, 2 TaggedDocuments, and
             a dataframe as its members
    DETAILS: The 2 Doc2Vec models can be accessed by pv.model_content_dbow and pv.model_title_dbow.
             These are committed to disk when build_doc2vec_content_model and build_doc2vec_title_model are
              called (as Embeddings/doc2vec_dbow_model_content and Embeddings/doc2vec_dbow_model_title resp.)"""
    pv = ParagraphVectorModel(df, sem_eval_dir_path=sem_eval_dir_path)
    # Remove df to save memory
    del df
    # Get docs of form [Word list, tag]: title and content tagged separately
    pv.get_tagged_docs()
    # Each of the models created in the foll. statemw
    pv.build_doc2vec_content_model()
    pv.build_doc2vec_title_model()
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
    parser.add_argument("--skip", '-s', action="store_true", default="False",
                        help="Use this argument to skip training the ML model")
    args = parser.parse_args()
    sem_eval_dir_path = args.path
    filename = '{}/data/IntegratedFiles/buzzfeed_training.tsv'.format(sem_eval_dir_path)
    df = clean_shuffle.read_prepare_df(filename)
    pv = build_pv_models(df, sem_eval_dir_path)

    if args.skip is False:
        # Get a composite embedding model
        X_train, y_train = get_vector_label_mapping(pv)
        svc = train_ml_model(X_train, y_train)
        # Serialize the model and save to disk
        joblib.dump(svc, '{}/models/svc_embeddings.joblib'.format(sem_eval_dir_path))

    print("DONE!")

if __name__ == '__main__':
    main()