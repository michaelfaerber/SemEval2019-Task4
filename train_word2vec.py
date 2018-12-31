from gensim.models import Word2Vec
import clean_shuffle
import os
import pandas as pd
import argparse
from nltk import word_tokenize, sent_tokenize
import numpy as np
import multiprocessing

cores = multiprocessing.cpu_count()
sem_eval_path = ''
train_val_boundary = 0.8

def load_texts(crowdsourced=False, split=True):
    tsv_name = 'crowdsourced_train' if crowdsourced is True else 'buzzfeed_training'
    df_name = 'crowdsourced_train_df' if crowdsourced is True else 'training_df'

    filename = os.path.join(sem_eval_path, 'data', 'IntegratedFiles', '{}_withid.tsv'.format(tsv_name))
    df_location = os.path.join(sem_eval_path, 'data', 'Pickles', '{}.pickle'.format(df_name))

    df = clean_shuffle.read_prepare_df(filename, file_path=df_location)

    df["text"] = df["title"] + ' ' + df["content"]

    return df

def load_merged_texts(crowdsourced, mix=True):
    train_df = load_texts(crowdsourced=crowdsourced, split=False)
    val_df = load_texts(crowdsourced=crowdsourced, split=False)

    base_path = os.path.join(sem_eval_path, 'data', 'Pickles')
    train_df_name = 'mixed_training' if mix is True else 'training'
    val_df_name = 'mixed_validation' if mix is True else 'validation'
    train_df_path = os.path.join(base_path, '{}_df.pickle'.format(train_df_name))
    val_df_path = os.path.join(base_path, '{}_df.pickle'.format(val_df_name))

    print('Loading dataframes from disk')
    train_df = pd.read_pickle(train_df_path)
    val_df = pd.read_pickle(val_df_path)

    print('Dataframes shapes:')
    print(train_df.shape)
    print(val_df.shape)

    return train_df['text']


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
  parser.add_argument("--crowdsourced", '-c', action='store_true', default="False",
                        help="Use this argument to work with the crowdsourced file")
  args = parser.parse_args()

  global sem_eval_path
  sem_eval_path = args.path

  crowdsourced = args.crowdsourced

  texts = load_merged_texts(crowdsourced)
  
  sentences = []

  for article in texts:
    sent_text = sent_tokenize(article)
    for sentence in sent_text:
      words = word_tokenize(sentence)
      sentences.append(words)
  
  model = Word2Vec(sentences, min_count=1, workers=cores)
  print(model)
  print(model['trump'])
  
  dataset_name = 'crowdsourced_' if crowdsourced is True else ''
  path = os.path.join(sem_eval_path, 'models', '{}words2vec.bin'.format(dataset_name))
  model.save(path)
  
  new_model = Word2Vec.load(path)
  print(new_model)

if __name__ == "__main__":
    main()