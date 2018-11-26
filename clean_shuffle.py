#------------------------------------------------------------------------------------
# Name:        Clean and shuffle data from input file and 
# Purpose:     This module is used create a pandas dataframe from the input
#              file and to clean it in various ways.
#
# Execution:   Not executable as a standalone program (collection of functions)
#
# Author:      Ashwath Sampath
#
# Created:     22-11-2018 (V1.0): Moved from common program to a separate module
# Revisions:   
#------------------------------------------------------------------------------------


import pandas as pd
import re
#from nltk.corpus import stopwords
import csv
from gensim.parsing import preprocessing
import contractions
# import inflect
from sklearn.utils import shuffle
from tqdm import tqdm

def clean_text(text):
    """ Cleans the text in the only argument in various steps 
    ARGUMENTS: text: content/title, string
    RETURNS: cleaned text, string"""
    # Replace newlines by space. We want only one doc vector.
    text = text.replace('\n', ' ').lower()
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Remove stop words
    text = preprocessing.remove_stopwords(text)
    # Remove html tags and numbers: can numbers possible be useful?
    text = preprocessing.strip_tags(preprocessing.strip_numeric(text))
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
    #text = re.sub(r'[^\w\s]', '', text.lower())   
    # STEMMING (Porter) automatically lower-cases as well
    # To stem or not to stem, that is the question
    #text = preprocessing.stem_text(text)
    return text

def read_prepare_df(filename):
    """ Read a file, put it in a dataframe. Drop unnecessary columns, clean the content.
    Please provide an absolute path.
    ARGUMENTS: filename: path to the input file, string
    RETURNS: df: a 'cleaned' Pandas dataframe with 3 columns (content, title and hyperpartisan) in
                 which nulls in content/title have been dropped"""
    df = pd.read_csv(filename, sep='\t', encoding='utf-8', names=['title','content','hyperpartisan'])
    print("Original DF shape = {}".format(df.shape))
    # url is useless, remove it. Remove bias too, and id. I no longer have them.
    #df = df.drop(['id', 'url', 'bias'], axis=1)
    # Drop NaNs!!
    df = df[pd.notnull(df['content'])]
    df = df[pd.notnull(df['title'])]
    # Question: should I combine the title and content in one field?

    df.content = df['content'].apply(clean_text)
    df.title = df['title'].apply(clean_text)
    # Shuffle it
    df = shuffle(df, random_state=13)
    print("Dataframe shape after cleaning = {}".format(df.shape))
    return df
