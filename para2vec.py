#------------------------------------------------------------------------------------
# Name:        Para2Vec
# Purpose:     This module contains the main ParagraphVectorModel class which is used
#              to build 2 doc2vec dbow models. Docs are tagged to their hyperpartisan
#              indicator (0/1). It also includes functions to combine
#              embeddings and to map document vectors to their original labels.
#
# Execution:   Not executable
#
# Author:      Ashwath Sampath
#
# Created:     24-11-2018 (V1.0): Class which builds 2 dbow models (which are 
#                                 (internally committed to disk). 
#                                 Functions which combine embeddings and map doc
#                                 embeddings with their original labels.
#                                 
# Revisions:   
#------------------------------------------------------------------------------------

import pandas as pd
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from tqdm import tqdm
import logging

logging.basicConfig(filename='/home/ashwath/Files/SemEval/logs/info_log.log', filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class ParagraphVectorModel:
    """ Creates a set of paragraph vectors"""
    def __init__(self, df, init_models=True):
        """ ARGUMENTS: df: a dataframe with 3 columns -- title, content and hyperpartisan"""
        self.df = df
        self.tagged_contentdocs = pd.Series()
        self.tagged_titledocs = pd.Series()

        # if init_models is False, it is a validation/test set, and the models should not be 
        # initialized here.
        if init_models:
            # Initialize doc2vec models without sentences (with negative sampling).
            # Downsample the content model
            self.model_content_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=5,
                                              workers=64, epochs=20, sample=1e-4, window=5)
            # Interleave skip gram vectors in the title model
            self.model_title_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=3,
                                            workers=64, epochs=20, sample=0, dbow=1)

    def get_tagged_docs(self, title_minlen=2, content_minlen=3):
        """ Gets tagged docs for a training/validation/test dataframe (both content and title
         are tagged with 'hyperpartisan' (0 or 1)).
        ARGUMENTS: title_minlen: the min length words which are to be retained in the TaggedDocument
                                 for the title words (default 2), int
                   content_minlen: the min length words which are to be retained in the TaggedDocument
                                   for the content words (default 3), int
        CLASS VARIABLES SET: tagged_titledocs: pd Series containing TaggedDocuments. Each list of words in the titles is
                                               mapped to its label (title and labels are from a pd Dataframe) 
                             tagged_contentdocs: pd Series containing TaggedDocuments. Each list of words in the content
                                                 is mapped to its label (content and labels are from a pd Dataframe) 
        RETURNS: None
        """     
        # Tokenize! 
        self.tagged_titledocs = self.df.apply(lambda row: TaggedDocument(
            words=simple_preprocess(row.title, deacc=True, min_len=2), tags=[row.hyperpartisan]), axis=1)
        self.tagged_contentdocs = self.df.apply(lambda row: TaggedDocument(
            words=simple_preprocess(row.content, deacc=True, min_len=3), tags=[row.hyperpartisan]), axis=1)

    def build_doc2vec_content_model(self):
        """ Trains a DBOW doc2vec model on the content after building the vocabulary
        ARGUMENTS: None
        CLASS VARIABLES SET: model_content_dbow, Doc2Vec model (MODEL SAVED TOO)
        RETURNS: None
        """  
        self.model_content_dbow.build_vocab(self.tagged_contentdocs.values)
        # Training time!
        self.model_content_dbow.train(tqdm(self.tagged_contentdocs.values),
                                      total_examples=self.model_content_dbow.corpus_count,
                                      epochs=self.model_content_dbow.epochs)

        fname = get_tmpfile("/home/ashwath/Files/SemEval/embeddings/doc2vec_dbow_model_content")
        self.model_content_dbow.save(fname)
        # TRAINING IS DONE: REMOVE THE MODEL TO SAVE MEMORY (infer_vector is still possible
        # as the vectors are present) 
        self.model_content_dbow.delete_temporary_training_data(keep_doctags_vectors=True,
                                                  keep_inference=True)

    def build_doc2vec_title_model(self):
        """ Trains a DBOW doc2vec model on the title after building the vocabulary
        ARGUMENTS: None
        CLASS VARIABLES SET: model_title_dbow, Doc2Vec model (MODEL SAVED TOO)
        RETURNS: None
        """  
        self.model_title_dbow.build_vocab(self.tagged_titledocs.values)
        # Training time!
        self.model_title_dbow.train(tqdm(self.tagged_titledocs.values),
                                    total_examples=self.model_title_dbow.corpus_count,
                                    epochs=self.model_title_dbow.epochs)

        fname = get_tmpfile("/home/ashwath/Files/SemEval/embeddings/doc2vec_dbow_model_title")
        self.model_title_dbow.save(fname)
        # TRAINING IS DONE: REMOVE THE MODEL TO SAVE MEMORY (infer_vector is
        # still possible as the vectors are present)
        # We don't need the whole model as we aren't going to train again.
        # https://radimrehurek.com/gensim/models/doc2vec.html
        self.model_title_dbow.delete_temporary_training_data(keep_doctags_vectors=True,
                                                        keep_inference=True)

    def content_vectors_and_labels(self):
        """ Infer the vectors of all the documents in the tagged documents using the model sent as a parameter. 
        ARGUMENTS: None
        RETURNS: para_vectors: tuple of 300-dimensional numpy arrays for each title
                 labels: corresponding labels in a Pandas Series
        """
        documents = self.tagged_contentdocs.values
        # Unzip into vectors and labels, and return them
        labels, para_vectors = zip(*[[doc.tags[0], self.model_content_dbow.infer_vector(doc.words)] for doc in documents])
        return para_vectors, pd.Series(labels)

    def title_vectors_and_labels(self):
        """ Infer the vectors of all the documents in the tagged documents using the model sent as a parameter.
        ARGUMENTS: None
        RETURNS: para_vectors: tuple of 300-dimensional numpy arrays for each title
                 labels: corresponding labels in a Pandas Series
        """
        # tagged_documents is a Pandas series, convert it into a numpy array of Tagged Documents
        titles = self.tagged_titledocs.values
        # Unzip into vectors and labels, and return them
        labels, para_vectors = zip(*[[title.tags[0], self.model_title_dbow.infer_vector(title.words)] for title in titles])
        return para_vectors, pd.Series(labels)

def add_embeddings(X1, X2):
    """ Adds 2 sets of embeddings (title + content embeddings). X1 and X2 are lists of numpy arrays,
     which must be added element-wise.
    ARGUMENTS: X1: Tuple of numpy arrays where each array is a 300-dimensional vector
               X2: Tuple of numpy arrays where each array is a 300-dimensional vector
    RETURNS: result, Tuple of numpy arrays where each array is a 300-dimensional vector. Each
                     of these arrays is obtained by adding the corresponding arrays in X1 and X2
     """
    # X1.shape should be equal to X2.shape.
    assert (X1[0].shape == X2[0].shape), "Content and title embeddings are of different shapes"
    result = [X1[i]+X2[i] for i in range(len(X1))]
    return result

def average_embeddings(X1, X2):
    """ Adds 2 sets of embeddings (title + content embeddings). X1 and X2 are lists of numpy arrays,
    which must be added element-wise
    ARGUMENTS: X1: Tuple of numpy arrays where each array is a 300-dimensional vector
               X2: Tuple of numpy arrays where each array is a 300-dimensional vector
    RETURNS: result, Tuple of numpy arrays where each array is a 300-dimensional vector. Each
                     of these arrays is obtained by averaging the corresponding arrays in X1 and X2
    """
    # X1.shape should be equal to X2.shape.
    assert (X1[0].shape == X2[0].shape), "Content and title embeddings are of different shapes"
    result = [(X1[i]+X2[i])/2 for i in range(len(X1))]
    return result

def concatenate_embeddings(X1, X2):
    """ Concatenates 2 sets of embeddings (title + content embeddings). X1 and X2 are lists of numpy arrays,
     which must be concatenated
    ARGUMENTS: X1: Tuple of numpy arrays where each array is a 300-dimensional vector
               X2: Tuple of numpy arrays where each array is a 300-dimensional vector
    RETURNS: result, Tuple of numpy arrays where each array is a 600-dimensional vector. Each
                     of these arrays is obtained by concatenating the corresponding arrays in X1 and X2"""
    result = [np.concatenate((X1[i], X2[i])) for i in range(len(X1))]
    return result

def get_vector_label_mapping(pv, method='avg'):
    """ Function which obtains the vector-label mapping for both the title and the content and returns
    composite matrix made up of vectors formed by the method specified in 'method'
    ARGUMENTS: pv: a ParagraphVectorModel instance consisting of 2 Doc2vec dbow models
               method: one of 'avg', 'concat' and 'sum' (default is 'avg'). Method by which a composite
                       vector is formed from the title and content vectors of each training .
    RETURNS: X_composite: a single matrix which is made up of combined title and content vectors
             y: Labels associated with the vectors
        """
    # Get vector-label mapping for content and title: both are in the same order as the shuffling was done
    #  before they were split. y_title = y_content
    X_content, y_content = pv.content_vectors_and_labels()
    X_title, y_title = pv.title_vectors_and_labels()
    if method == 'avg':
        X_composite = average_embeddings(X_content, X_title)
    if method == 'sum':
        X_composite = add_embeddings(X_content, X_title)
    if method == 'concat':
        X_composite = concatenate_embeddings(X_content, X_title)
    return X_composite, y_content