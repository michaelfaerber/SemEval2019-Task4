import ground_truth_sqlite
import clean_shuffle
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import pickle
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from numpy import zeros

class TextsLoader:
  def __init__(self, sem_eval_path, crowdsourced, logs_path, train_val_boundary=0.8):
    self.sem_eval_path = sem_eval_path
    self.crowdsourced = crowdsourced
    self.train_val_boundary = train_val_boundary
    logging.basicConfig(filename=logs_path, filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

  def load(self, split=True, validation=False):
      name = 'validation' if validation else 'training'
      tsv_name = 'crowdsourced_train' if self.crowdsourced is True else 'buzzfeed_{}'.format(name)
      table_name = 'crowdsourced_train' if self.crowdsourced is True else name
      df_name = 'crowdsourced_train_df' if self.crowdsourced is True else '{}_df'.format(name)

      filename = os.path.join(self.sem_eval_path, 'data', 'IntegratedFiles', '{}_withid.tsv'.format(tsv_name))
      df_location = os.path.join(self.sem_eval_path, 'data', 'Pickles', '{}.pickle'.format(df_name))

      df = clean_shuffle.read_prepare_df(filename, file_path=df_location)

      ids_to_labels = ground_truth_sqlite.select_id_hyperpartisan_mappings(self.sem_eval_path, 'ground_truth_{}'.format(table_name))
      df['hyperpartisan'] = df.apply(lambda row: 1 if ids_to_labels[row['id']] == 'true' else 0, axis=1)

      df["text"] = df["title"] + ' ' + df["content"]

      if split:
          boundary = int(self.train_val_boundary * df['text'].shape[0])
          return df['text'][:boundary], df['hyperpartisan'][:boundary], df['text'][boundary:], df['hyperpartisan'][boundary:]
      else:
          return df

  def load_mixed(self, use_3_sets=False):
      train_df = self.load(split=False, validation=False)
      val_df = self.load(split=False, validation=True)

      base_path = os.path.join(self.sem_eval_path, 'data', 'Pickles')
      train_df_path = os.path.join(base_path, 'mixed_training_df.pickle')
      val_df_path = os.path.join(base_path, 'mixed_validation_df.pickle')

      if os.path.isfile(train_df_path) and os.path.isfile(val_df_path):
          logging.info('Loading mixed dataframes from disk')
          train_df = pd.read_pickle(train_df_path)
          val_df = pd.read_pickle(val_df_path)
      else:
          logging.info('Creating new mixed dataframes')

          df = self._mix_datasets(train_df, val_df)

          # Split train/test
          if use_3_sets:
            train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['hyperpartisan'], random_state=1)
            train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=df['hyperpartisan'], random_state=1)
            return train_df['text'], train_df['hyperpartisan'], val_df['text'], val_df['hyperpartisan'], test_df['text'], test_df['hyperpartisan']
          else:
            train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['hyperpartisan'])
            return train_df['text'], train_df['hyperpartisan'], val_df['text'], val_df['hyperpartisan']

  def _mix_datasets(self, train_df, val_df):
    # Append
    df = train_df.append(val_df, ignore_index=True)
    logging.info('Appended shape: {}'.format(df.shape))
    logging.info(df[:2])

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    logging.info(df[:2])

    return df

class TokenizerLoader:
    def __init__(self, texts, sem_eval_path, logs_path, most_common_count=100000):
        self.texts = texts
        self.sem_eval_path = sem_eval_path
        self.num_words = most_common_count + 1
        self.file_path = os.path.join(self.sem_eval_path, 'data', 'Tokenizers', 'buzzfeed_trained_{}_tokenizer.pickle'.format(self.num_words))
        logging.basicConfig(filename=logs_path, filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def load(self):
        if os.path.isfile(self.file_path):
            return self._load_trained_tokenizer()
        else:
            return self._train_tokenizer()

    def _load_trained_tokenizer(self):
        with open(self.file_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
        logging.info('Tokenizer loaded from disk')
        return tokenizer

    def _train_tokenizer(self):
        tokenizer = Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(self.texts)

        with open(self.file_path, 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Tokenizer fit on texts and stored on disk')

        return tokenizer

class WordVectorsLoader:
    def __init__(self, sem_eval_path, crowdsourced, embedding_mode):
        self.sem_eval_path = sem_eval_path
        self.crowdsourced = crowdsourced
        self.embedding_mode = embedding_mode
        self.embedding_dims = 300
        self.word_vectors = {}

    def load(self):
        if self.embedding_mode == 0:
            self.word_vectors = self._load_Google_word2vec()
        elif self.embedding_mode == 1:
            self.word_vectors = self._load_Glove()
        elif self.embedding_mode == 2:
            raise Exception('FastText not implemented yet')
        elif self.embedding_mode == 3:
            self.word_vectors = self._load_custom_pretrained()
        else:
            raise Exception('Unknown input for embedding_mode.')

    def create_embedding_weights_matrix(self, word_index):
        weights_matrix = zeros((len(word_index) + 1, self.embedding_dims))

        count = 0
        for word, idx in word_index.items():
            if word in self.word_vectors:
                weights_matrix[idx] = self.word_vectors[word]
                count += 1
        logging.info('Words found on word2vec: {}'.format(count))

        return weights_matrix 

    def _load_Google_word2vec(self):
        logging.info("Loading Google's word2vec vectors")
        self.embedding_dims = 300
        filename = '{}/GoogleNews-vectors-negative300.bin'.format(self.sem_eval_path)
        model = KeyedVectors.load_word2vec_format(filename, binary=True)
        return model.wv

    def _load_Glove(self):
        logging.info("Loading Glove word vectors")
        self.embedding_dims = 100
        word_index = {}
        f = open(os.path.join(self.sem_eval_path, 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            word_index[word] = coefs
        f.close()
        logging.info('Found %s word vectors.' % len(word_index))
        return word_index

    def _load_custom_pretrained(self):
        logging.info("Loading custom pretrained word2vec vectors")
        self.embedding_dims = 100
        dataset_name = 'crowdsourced_' if self.crowdsourced is True else ''
        path = os.path.join(self.sem_eval_path, 'models', '{}words2vec.bin'.format(dataset_name))
        model = Word2Vec.load(path)
        return model.wv

class TextSequencesLoader:
    def __init__(self, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def load(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        # Free up memory
        del texts
        with tf.device('/cpu:0'):
            X_train = pad_sequences(sequences, maxlen=self.seq_len, padding='post')
            del sequences
        return X_train
