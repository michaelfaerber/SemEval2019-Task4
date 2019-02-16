import ground_truth_sqlite
import clean_shuffle
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import pickle
from gensim.models import KeyedVectors, Word2Vec, FastText
import numpy
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

pd.set_option('display.float_format', lambda x: '%.5f' % x)

class TextsLoader:
  def __init__(self, sem_eval_path, crowdsourced, logs_path='', train_val_boundary=0.8):
    self.sem_eval_path = sem_eval_path
    self.crowdsourced = crowdsourced
    self.train_val_boundary = train_val_boundary
    if not logs_path:
        logs_path = os.path.join(sem_eval_path, 'logs', 'test_log.log')
    logging.basicConfig(filename=logs_path, filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

  def load(self, split=False, validation=False, sentences=False):
      name = 'validation' if validation else 'training'
      tsv_name = 'crowdsourced_train' if self.crowdsourced is True else 'buzzfeed_{}'.format(name)
      table_name = 'crowdsourced_train' if self.crowdsourced is True else name
      sentences_phrase = '_s' if sentences else ''
      df_name = 'crowdsourced_train{}_df'.format(sentences_phrase) if self.crowdsourced is True else '{}{}_df'.format(name, sentences_phrase)

      filename = os.path.join(self.sem_eval_path, 'data', 'IntegratedFiles', '{}_withid.tsv'.format(tsv_name))
      df_location = os.path.join(self.sem_eval_path, 'data', 'Pickles', '{}.pickle'.format(df_name))

      print('DataFrame file location: {}'.format(df_location))
      if sentences:
        df = clean_shuffle.read_prepare_sentence_df(filename, file_path=df_location)
      else:
        df = clean_shuffle.read_prepare_df(filename, file_path=df_location)

      print('df {} loaded. Shape: {}'.format(df_name, df.shape))

      ids_to_labels = ground_truth_sqlite.select_id_hyperpartisan_mappings(self.sem_eval_path, 'ground_truth_{}'.format(table_name))
      df['hyperpartisan'] = df.apply(lambda row: 1 if ids_to_labels[row['id']] == 'true' else 0, axis=1)

      if not 'text' in df.columns:
        df["text"] = df["title"] + ' ' + df["content"]

      if split:
          boundary = int(self.train_val_boundary * df['text'].shape[0])
          return df['text'][:boundary], df['hyperpartisan'][:boundary], df['text'][boundary:], df['hyperpartisan'][boundary:]
      else:
          return df['text'], df['hyperpartisan']

  def load_mixed(self, use_3_sets=False, sentences=False):
      base_path = os.path.join(self.sem_eval_path, 'data', 'Pickles')
      train_df_path = os.path.join(base_path, 'mixed{}_training_df.pickle'.format('_s' if sentences else ''))
      val_df_path = os.path.join(base_path, 'mixed{}_validation_df.pickle'.format('_s' if sentences else ''))

      logging.info('train_df_path: {}'.format(train_df_path))
      logging.info('val_df_path: {}'.format(val_df_path))

      if os.path.isfile(train_df_path) and os.path.isfile(val_df_path):
          train_df = pd.read_pickle(train_df_path)
          val_df = pd.read_pickle(val_df_path)
          return train_df['text'], train_df['hyperpartisan'], val_df['text'], val_df['hyperpartisan']
      else:
          logging.info('Creating new mixed dataframes')

          train_df = self.load(split=False, validation=False, sentences=sentences)
          val_df = self.load(split=False, validation=True, sentences=sentences)

          df = self._mix_datasets(train_df, val_df)

          # Split train/test
          if use_3_sets:
            train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['hyperpartisan'], random_state=1)
            train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=df['hyperpartisan'], random_state=1)
            return train_df['text'], train_df['hyperpartisan'], val_df['text'], val_df['hyperpartisan'], test_df['text'], test_df['hyperpartisan']
          else:
            logging.info('Splitting new mixed dataframes')
            train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['hyperpartisan'])
            logging.info('Storing new mixed dataframes to disk')
            train_df.to_pickle(train_df_path)
            val_df.to_pickle(val_df_path)
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
    def __init__(self, texts, sem_eval_path, logs_path='', most_common_count=100000):
        self.texts = texts
        self.sem_eval_path = sem_eval_path
        self.num_words = most_common_count + 1
        self.file_path = os.path.join(self.sem_eval_path, 'data', 'Tokenizers', 'buzzfeed_trained_{}_tokenizer.pickle'.format(self.num_words))
        if logs_path:
            logging.basicConfig(filename=logs_path, filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def load(self):
        if os.path.isfile(self.file_path):
            return self._load_trained_tokenizer()
        else:
            logging.info('Training new tokenizer...')
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
            self.word_vectors = self._load_fasttext()
        elif self.embedding_mode == 3:
            self.word_vectors = self._load_custom_pretrained_word2vec()
        elif self.embedding_mode == 4:
            self.word_vectors = self._load_custom_pretrained_fasttext()
        elif self.embedding_mode == 5:
            self.word_vectors = self._load_custom_pretrained_news_word2vec()
        else:
            raise Exception('Unknown input for embedding_mode.')

    def create_embedding_weights_matrix(self, word_index):
        weights_matrix = numpy.zeros((len(word_index) + 1, self.embedding_dims))

        count = 0
        for word, idx in word_index.items():
            if word in self.word_vectors:
                weights_matrix[idx] = self.word_vectors[word]
                count += 1
        logging.info('Words found on word2vec: {}'.format(count))

        return weights_matrix 

    def _load_fasttext(self):
        logging.info("Loading FastText word vectors")
        self.embedding_dims = 300
        filename = '{}/wiki-news-300d-1M-subword.vec'.format(self.sem_eval_path)
        model = KeyedVectors.load_word2vec_format(filename)
        return model.wv

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

    def _load_custom_pretrained_word2vec(self):
        logging.info("Loading custom pretrained word2vec vectors")
        self.embedding_dims = 100
        dataset_name = 'crowdsourced_' if self.crowdsourced is True else ''
        path = os.path.join(self.sem_eval_path, 'models', '{}words2vec.bin'.format(dataset_name))
        model = Word2Vec.load(path)
        return model.wv

    def _load_custom_pretrained_fasttext(self):
        logging.info("Loading custom pretrained fasttext vectors")
        self.embedding_dims = 100
        dataset_name = 'crowdsourced_' if self.crowdsourced is True else ''
        path = os.path.join(self.sem_eval_path, 'models', '{}custom_fasttext.bin'.format(dataset_name))
        model = FastText.load(path)
        return model.wv

    def _load_custom_pretrained_news_word2vec(self):
        logging.info("Loading custom pretrained news word2vec vectors")
        self.embedding_dims = 100
        path = os.path.join(self.sem_eval_path, 'models', 'news_texts', 'news_words2vec_cbow.bin')
        model = Word2Vec.load(path)
        return model.wv

class TextSequencesLoader:
    def __init__(self, tokenizer, seq_len, sem_eval_path=''):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.indices_to_remove = []
        logs_path = os.path.join(sem_eval_path, 'logs', 'sequences_distribution.log')
        logging.basicConfig(filename=logs_path, filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def load(self, texts, truncate_sequences=False):
        self.indices_to_remove = []
        sequences = self.tokenizer.texts_to_sequences(texts)
        sen_lengths = numpy.array([len(item) for item in sequences])
        logging.info('Seq Min: {}'.format(sen_lengths.min()))
        logging.info('Seq Avg: {}'.format(sen_lengths.mean()))
        logging.info('Seq Std: {}'.format(sen_lengths.std()))
        logging.info('Seq Max: {}'.format(sen_lengths.max()))
        gt_5000_count = len([length for length in sen_lengths if length > 5000])
        logging.info('Count > 5000: {} from {}'.format(gt_5000_count, len(sen_lengths)))
        gt_2500_count = len([length for length in sen_lengths if length > 2500])
        logging.info('Count > 2500: {} from {}'.format(gt_2500_count, len(sen_lengths)))
        
        # Free up memory
        del texts

        # Mark to remove the empty sequences
        for idx, item in enumerate(sequences):
            if len(item) == 0:
                self.indices_to_remove.append(idx)

        if truncate_sequences:
            self._truncate_sequences(sequences)
            
        # Pad sequences using CPU because of memory issues
        with tf.device('/cpu:0'):
            X_values = pad_sequences(sequences, maxlen=self.seq_len, padding='post')
            del sequences
        return X_values
    
    def _truncate_sequences(self, sequences):
        sen_lengths = numpy.array([len(item) for item in sequences])
        logging.info('Min: {}'.format(sen_lengths.min()))
        logging.info('Avg: {}'.format(sen_lengths.mean()))
        logging.info('Std: {}'.format(sen_lengths.std()))
        logging.info('Max: {}'.format(sen_lengths.max()))
        self.seq_len = int(sen_lengths.mean() + 4 * sen_lengths.std())
        logging.info('New seq length (mean + 4 * std): {}'.format(self.seq_len))
        logging.info('Sentences to be truncated: {}'.format(len([length for length in sen_lengths if length > self.seq_len])))
