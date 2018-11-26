import pandas as pd
import re
#from nltk.corpus import stopwords
import gensim
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix, classification_report
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from gensim.parsing import preprocessing
import contractions
# import inflect
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import logging


#NOTE: doctag_syn0 gets the weights for each tag in the tagged document. 2 tags -> It'll be len=2
 
logging.basicConfig(filename='/home/ashwath/Files/SemEval/info_log.log', filemode='w', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
results_log = open('/home/ashwath/Files/SemEval/finished_log.log', 'a')

def clean_text(text):
    """ Cleans the text in various steps """
    # Replace newlines by space. We want only one doc vector.
    text = text.replace('\n', ' ').lower()
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Remove stop words and short words (<3 char)
    text = preprocessing.remove_stopwords(text)
    # Remove html tags and numbers: can numbers possible be useful?
    # Inflect can convert numbers to words: https://pypi.org/project/inflect/
    # https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
    text = preprocessing.strip_tags(preprocessing.strip_numeric(text))
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
    #text = re.sub(r'[^\w\s]', '', text.lower())   
    # STEMMING (Porter) automatically lower-cases as well
    # To stem or not to stem, that is the question
    #text = preprocessing.stem_text(text)
    # Add a space to the start so that title can be combined with sentence later
    #text = ' ' + text
    return text

def read_prepare_df(filename):
    """ Read a file, put it in a dataframe. Drop unnecessary columns, clean the content"""
    df = pd.read_csv(filename, sep='\t', encoding='utf-8', names=['title','content','hyperpartisan'])
    # url is useless, remove it. Remove bias too, and id. I no longer have them.
    #df = df.drop(['id', 'url', 'bias'], axis=1)
    # Drop NaNs!!
    print(df.columns)
    df = df[pd.notnull(df['content'])]
    df = df[pd.notnull(df['title'])]
    # Question: should I combine the title and content in one field?

    df.content = df['content'].apply(clean_text)
    df.title = df['title'].apply(clean_text)
    # Shuffle it
    df = shuffle(df, random_state=13)
    return df

# Add title to the content
# df.content = df.title.str.cat(df.content)

def build_doc2vec_model(tagged_docs):
    """ Trains a DBOW doc2vec model after building the vocabulary"""  
    # DBOW, vector size=300, min_count -> slightly high value, workers = num_cores. sample = 0 by default. Negative sampling rather than hierarchical softmax
    # window-size may be a candidate for changing.
    #model_dbow = Doc2Vec(dm=0, vector_size=300, dbow_words=1, negative=5, hs=0, min_count=10, workers=64, epochs=20, sample=0)
    # Downsample as there are too many words
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=5, workers=64, epochs=20, sample=1e-4, window=5)
    model_dbow.build_vocab(tagged_docs.values)
    # Training time!
    model_dbow.train(tqdm(tagged_docs.values), total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

    fname = get_tmpfile("/home/ashwath/Files/SemEval/doc2vec_dbow_model")
    model_dbow.save(fname)
    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return model_dbow
    model_dm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5)
    # TRAINING IS DONE: REMOVE THE MODEL TO SAVE MEMORY (infer_vector is still possible as the vectors are present) 
    # https://radimrehurek.com/gensim/models/doc2vec.html
    model_dm.build_vocab(tagged_docs.values)
    # Training time!
    model_dm.train(tqdm(tagged_docs.values), total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
    model_dm.save("/home/ashwath/Files/SemEval/doc2vec_dm_model")
    model_dm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_dbowdm = ConcatenatedDoc2Vec([model_dbow, model_dm])
    fname = get_tmpfile("/home/ashwath/Files/SemEval/doc2vec_dbowdm_model")
    #model_dbowdm.save(fname)
    #model_dbowdm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    #return model_dbowdm

def build_doc2vec_title_model(tagged_docs):
    """ Trains a DBOW doc2vec model after building the vocabulary (for title)"""  
    # DBOW, vector size=300, min_count -> slightly high value, workers = num_cores. sample = 0 by default. Negative sampling rather than hierarchical softmax
    # window-size may be a candidate for changing.
    #model_dbow = Doc2Vec(dm=0, vector_size=300, dbow_words=1, hs=1 (hierarchical softmax), min_count=10, workers=64, epochs=20, sample=0)
    # Also interleave word vectors (skipgram). But no downsampling for the title
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=3, workers=64, epochs=20, sample=0, dbow=1)
    model_dbow.build_vocab(tagged_docs.values)
    # Training time!
    model_dbow.train(tqdm(tagged_docs.values), total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

    fname = get_tmpfile("/home/ashwath/Files/SemEval/doc2vec_dbow_model_title")
    model_dbow.save(fname)
    # TRAINING IS DONE: REMOVE THE MODEL TO SAVE MEMORY (infer_vector is still possible as the vectors are present)
    # We don't need the whole model as we aren't going to train again.
    # https://radimrehurek.com/gensim/models/doc2vec.html
    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return model_dbow

def get_vectors_labels(model, tagged_documents):
    """ Infer the vectors of all the documents in the tagged documents using the model sent as a parameter. """
    # tagged_documents is a Pandas series, convert it into a numpy array of Tagged Documents
    docs = tagged_documents.values
    labels, para_vectors = zip(*[[doc.tags[0], model.infer_vector(doc.words)] for doc in docs])
    return para_vectors, pd.Series(labels)

def train_model(X_train, y_train):
    """ Trains a machine learning model based on a set of embeddings. Returns the model"""
    svc = SGDClassifier(alpha=0.0001, loss = 'hinge', max_iter=1000, n_jobs=-1)
    svc.fit(X_train, y_train)
    return svc

def predict_vals(model, X_val, y_val):
    return pd.Series(model.predict(X_val))

def get_train_tagged_docs(train):
    """ Gets tagged docs for the training dataframe (both content and title)"""
    # Is removing 2-letter words dangerous? It'll remove of, in etc. But it will also remove names like 'Ed'. I think it's all right
    # Use simple_preprocess to tokenize while removing accents and words less than 3 characters long. It also lowercases words, so the clean_text part
    # isn't actually needed
    # tagged_train_docs is a Pandas series with words
    train_tagged_titledocs = train.apply(lambda row: TaggedDocument(words=simple_preprocess(row.title, deacc=True, min_len=2), tags=[row.hyperpartisan]), axis=1)
    train_tagged_docs = train.apply(lambda row: TaggedDocument(words=simple_preprocess(row.content, deacc=True, min_len=3), tags=[row.hyperpartisan]), axis=1)
    return train_tagged_docs, train_tagged_titledocs

def get_val_tagged_docs(validation):
    """ Gets tagged docs for the validation dataframe (both content and title)"""
    val_tagged_titledocs = validation.apply(lambda row: TaggedDocument(words=simple_preprocess(row.title, deacc=True, min_len=2), tags=[row.hyperpartisan]), axis=1)
    val_tagged_docs = validation.apply(lambda row: TaggedDocument(words=simple_preprocess(row.content, deacc=True, min_len=3), tags=[row.hyperpartisan]), axis=1)
    return val_tagged_docs, val_tagged_titledocs

def get_title_embeddings(train_tagged_titledocs, val_tagged_titledocs):
    """  Gets embeddings for the content by training a doc2vec dbow model """
    model_title_dbow = build_doc2vec_title_model(train_tagged_titledocs)
    X_title_train, y_title_train = get_vectors_labels(model_title_dbow, train_tagged_titledocs)
    X_title_val, y_title_val = get_vectors_labels(model_title_dbow, val_tagged_titledocs)
    return X_title_train, y_title_train, X_title_val, y_title_val

def get_content_embeddings(train_tagged_docs, val_tagged_docs):
    """ Gets embeddings for the content by training a doc2vec dbow model """
    model_dbow = build_doc2vec_model(train_tagged_docs)
    # Get a combination of vectors and labels
    X_train, y_train = get_vectors_labels(model_dbow, train_tagged_docs)
    # Infer vectors for test (val) set as well
    X_val, y_val = get_vectors_labels(model_dbow, val_tagged_docs)
    return X_train, y_train, X_val, y_val

def add_embeddings(X1, X2):
    """ Adds 2 sets of embeddings (title + content embeddings). X1 and X2 are lists of numpy arrays, which must be added element-wise
    (In deep learning terms, these are actually tensors)"""
    # X1.shape should be equal to X2.shape.
    result = [X1[i]+X2[i] for i in range(len(X1))]
    return result

def average_embeddings(X1, X2):
    """ Adds 2 sets of embeddings (title + content embeddings). X1 and X2 are lists of numpy arrays, which must be added element-wise
    (In deep learning terms, these are actually tensors)"""
    # X1.shape should be equal to X2.shape.
    result = [(X1[i]+X2[i])/2 for i in range(len(X1))]
    return result

def concatenate_embeddings(X1, X2):
    """ Concatenates 2 sets of embeddings (title + content embeddings). X1 and X2 are lists of numpy arrays, which must be concatenated
     element-wise (In deep learning terms, these are actually tensors)"""
    # X1.shape should be equal to X2.shape.
    result = [np.concatenate((X1[i], X2[i])) for i in range(len(X1))]
    return result

def train_predict_estimate(X_train, y_train, X_val, y_val, embeddings_method):
    """ Trains model 1 (only using the content embeddings) """ 
    ml_model = train_model(X_train, y_train)
    y_pred = predict_vals(ml_model, X_val, y_val)
    calculate_metrics(embeddings_method, y_train, y_val, y_pred, ml_model)

def calculate_metrics(embeddings_method, y_train, y_test, y_pred, ml_model):
    """ Calculates a number of metrics on the holdout set after training and getting the predictions."""
    results_log.write("Embeddings used: {}\n".format(embeddings_method))
    results_log.write("ML Model for classification: {}\n".format(ml_model))
    results_log.write("Predicted value counts per class (training set):\n {}\n".format(y_train.value_counts()))
    results_log.write("Predicted value counts per class (predictions):\n{}\n ".format(y_pred.value_counts()))
    results_log.write("Predicted value counts per class (val set):\n{}\n ".format(y_test.value_counts()))
    f1 = f1_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    accuracy = accuracy_score(y_test, y_pred)
    results_log.write("F1={}, Precision={}, Recall={}, Accuracy={}".format(f1, precision, recall, accuracy))
    results_log.write(classification_report(y_test, y_pred, target_names=['fair', 'biased'] ))
    results_log.write("Confusion matrix: \n{}\n".format(confusion_matrix(y_test, y_pred)))

def main():
    filename = '/home/ashwath/Files/SemEval/data/IntegratedFiles/buzzfeed_training.tsv'
    df = read_prepare_df(filename)
    results_log.write("Pandas takes memory: \n{}\n".format(df.info(memory_usage='deep')))
    # Train test split
    train, validation = train_test_split(df, shuffle=True, stratify=df.hyperpartisan, test_size=0.2, random_state=13)
    # Save memory, remove df
    del df
    #train = train[:2000]
    #validation = validation[:1000]
    # test docs will not have labels
    # test_corpus = test.apply(lambda row: simple_preprocess(row.content, deacc=True, min_len=3), axis=1)
    train_tagged_docs, train_tagged_titledocs = get_train_tagged_docs(train)
    del train
    val_tagged_docs, val_tagged_titledocs = get_val_tagged_docs(validation)
    del validation
    # Get embeddings after the model has been trained
    X_title_train, y_title_train, X_title_val, y_title_val = get_title_embeddings(train_tagged_titledocs, val_tagged_titledocs)
    X_train, y_train, X_val, y_val = get_content_embeddings(train_tagged_docs, val_tagged_docs)

    # Add title and content embeddings -> method 2 (y doesn't need to be touched)
    X_composite_train = average_embeddings(X_train, X_title_train)
    X_composite_val = average_embeddings(X_val, X_title_val)
    results_log.write('Embedding shapes => X_train = {}, y_train={}, X_val={}, y_val={}'.format(
        X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    
    # METHOD 1: Only content embeddings 
    embeddings_method1 = 'Method 1: Only using content embeddings (PV-DBOW with negative sampling and min_len=3)'
    print('****************************************************************************************************************\n')
    results_log.write('****************************************************************************************************************\n')
    train_predict_estimate(X_train, y_train, X_val, y_val, embeddings_method1)
    print('****************************************************************************************************************\n')
    results_log.write('****************************************************************************************************************\n')
    # METHOD 2: Content + title embeddings (vectors added)
    embeddings_method2 = """Method 2: Using content and title embeddings (Content: PV-DBOW with negative sampling and down sampling, min_len=2,
                         min_count=5;
                         Title: PV-DBOW with interleaved skipgram, hierarchical softmax, No down sampling, and min_len=2, min_count=5)"""
                         
    train_predict_estimate(X_composite_train, y_train, X_composite_val, y_val, embeddings_method2)
    results_log.close()

if __name__ == '__main__':
    main()

# 'clf', SGDClassifier(alpha=0.0001, loss = 'hinge', max_iter=1000))
'''
>>> import pandas as pd
>>> import gensim
>>> import csv
>>> from sklearn.model_selection import train_test_split
>>> from gensim.models.doc2vec import TaggedDocument
>>> from gensim.utils import simple_preprocess
>>> filename = '/home/ashwath/Files/SemEval/training_data.tsv'
>>> df = pd.read_csv(filename, sep='\t', encoding='utf-8')
>>> df = df.drop(['url', 'bias'], axis=1)
>>> def clean_text(text):
...     """ Makes text lower case, replaces \n with space """
...     text = text.lower()
...     # Replace newlines by space. We want only one doc vector.
...     text = text.replace('\n', ' ')
...     # Remove stop words??
...     return text
...
>>> df.content = df['content'].apply(clean_text)
>>> train, test = train_test_split(df, test_size=0.2, random_state=42)
>>> tagged_train_docs = train.apply(lambda row: TaggedDocument(words=simple_preprocess(row.content, deacc=True, min_len=3), tags=[row.hyperpartisan]), axis=1)
>>> type(tagged_train_docs)
'''

# model = Doc2Vec.load('/home/ashwath/Files/SemEval/doc2vec_dbow_model')
# If model is a Doc2Vec model, the word representations can be found by model['your_word_here'], or model.wv['your_word_here']. The doc representations
# (here there are only 2 classes: 0 and 1) are obtained by model.docvecs[0] and model.docvecs[1] (fair and hyperpartisan resp.). All these vectors are (300,0).
# It's the same for model2

# >>> model.syn1neg.shape
#(193127, 300)
# >>> model.wv.vectors.shape
# (193127, 300)

#model2 = Doc2Vec.load('/home/ashwath/Files/SemEval/doc2vec_dbow_model_title')
#>>> model2.wv.vectors.shape
#(37811, 300)
# >>> model.docvecs.doctag_syn0.shape (same for model2)
#(2, 300)
# doctag_syn0 are the input weights (between the hidden layer and the input layer), syn1neg are the output layer weights 
# (between hidden layer and ouptut layer with softmax)
# The normalized version goes in doctag_syn0norm

#   |  Data descriptors defined here:
# |
# |  doctag_syn0
# |
# |  doctag_syn0norm
# |
# |  index2entity

# https://groups.google.com/forum/#!topic/gensim/lyWt8d0X8fY
# https://groups.google.com/forum/#!msg/gensim/Fujja7aOH6E/C3WArofWbNIJ
# https://groups.google.com/forum/#!topic/gensim/RLRfY6k3ulw
#https://groups.google.com/forum/#!msg/gensim/JO0xBEh7gOY/Vf6ARlttFQAJ;context-place=msg/gensim/g3ZqXfXIaLA/plCIXhKZAwAJ