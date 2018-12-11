from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Activation, Embedding, Flatten
from keras.models import load_model
from gensim.models import KeyedVectors
import clean_shuffle
import argparse
from numpy import zeros
import pickle
import os.path
import tensorflow as tf
import nltk
import string

sem_eval_path = ''
dataset_type = ''
embedding_dims = 300
seq_len = 2500 # Inferred from checking the sequences length distributions
train_test_boundary = 0.8
most_common_count = 0



def load_word_vectors():
  print('Loading word vectors...')
  filename = '{}/GoogleNews-vectors-negative300.bin'.format(sem_eval_path)
  model = KeyedVectors.load_word2vec_format(filename, binary=True)
  return model.wv

def get_training_dataframe_path():
    basepath = '{}/data/Dataframes'.format(sem_eval_path)
    if dataset_type == 'training':
        return  '{}/buzzfeed_training.pkl'.format(basepath)
    elif dataset_type == 'crowdsourced_training':
        return  '{}/crowdsourced_train.pkl'.format(basepath)
    return 'Dummy: not yet available'

def get_testing_dataframe_path():
    basepath = '{}/data/Dataframes'.format(sem_eval_path)
    if dataset_type == 'training':
        return  '{}/buzzfeed_validation.pkl'.format(basepath)
    elif dataset_type == 'crowdsourced_training':
        return  '{}/buzzfeed_test.pkl'.format(basepath)
    return 'Dummy: not yet available'

def get_model_path():
  basepath = '{}/models/deep-learning'.format(sem_eval_path)
  if dataset_type == 'training':
      return  '{}/buzzfeed_word2vec_conv_model.h5'.format(basepath)
  elif dataset_type == 'crowdsourced_training':
      return  '{}/crowdsourced_word2vec_conv_model.h5'.format(basepath)
  return 'Dummy: not yet available'

def get_training_tsv_path():
  basepath = '{}/data/IntegratedFiles'.format(sem_eval_path)
  if dataset_type == 'training':
      return  '{}/buzzfeed_training.tsv'.format(basepath)
  elif dataset_type == 'crowdsourced_training':
      return  '{}/crowdsourced_train.tsv'.format(basepath)
  return 'Dummy: not yet available'

def get_testing_tsv_path():
  basepath = '{}/data/IntegratedFiles'.format(sem_eval_path)
  if dataset_type == 'training':
      return  '{}/buzzfeed_validation.tsv'.format(basepath)
  elif dataset_type == 'crowdsourced_training':
      return  '{}/buzzfeed_test.tsv'.format(basepath)
  return 'Dummy: not yet available'

def get_tokenizer_path(num_words):
    basepath = '{}/data/Tokenizers'.format(sem_eval_path)
    if dataset_type == 'training':
        return  '{}/buzzfeed_trained_{}_tokenizer.pickle'.format(basepath, num_words)
    elif dataset_type == 'crowdsourced_training':
        return  '{}/crowdsourced_trained_{}_tokenizer.pickle'.format(basepath, num_words)
    return 'Dummy: not yet available'

def get_common_words_texts_path(dataset_split):
    basepath = '{}/data/CommonWords'.format(sem_eval_path)
    if dataset_type == 'training':
        return  '{}/buzzfeed_{}_{}_common_words.pickle'.format(basepath, dataset_split, most_common_count)
    elif dataset_type == 'crowdsourced_training':
        return  '{}/crowdsourced_{}_{}_common_words.pickle'.format(basepath, dataset_split, most_common_count)
    return 'Dummy: not yet available'

def load_train_texts():
  train_data_file = get_training_tsv_path()
  train_dataframe_path = get_training_dataframe_path()
  train_df = clean_shuffle.read_prepare_df(train_data_file, file_path=train_dataframe_path)

  train_df["text"] = train_df["title"] + ' ' + train_df["content"]

  return train_df['text'], train_df['hyperpartisan']

def load_test_texts():
  test_data_file = get_testing_tsv_path()
  test_dataframe_path = get_testing_dataframe_path()
  test_df = clean_shuffle.read_prepare_df(test_data_file, file_path=test_dataframe_path)

  test_df["text"] = test_df["title"] + ' ' + test_df["content"]

  return test_df['text'], test_df['hyperpartisan']

def load_texts():
  X_train, y_train = load_train_texts()
  X_test, y_test = load_test_texts()
  return X_train, y_train, X_test, y_test

def load_tokenizer(X_train_texts):
    num_words = most_common_count + 1
    file_path = get_tokenizer_path(num_words)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        print('Tokenizer loaded from disk')
    else:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train_texts)

        with open(file_path, 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)

        print('Tokenizer fit on texts and stored on disk')
    return tokenizer

def get_words(text):
  tokens = nltk.tokenize.word_tokenize(text)
  
  table = str.maketrans('', '', string.punctuation)
  stripped = [w.translate(table) for w in tokens]
  # remove remaining tokens that are not alphabetic
  words = [word for word in stripped if word.isalpha()]

  # remove 1-character words
  words = [word for word in words if len(word) > 1]
  return words

def get_most_common_words(text):
  words = get_words(text)

  fdist = nltk.probability.FreqDist(words)
  top_words = fdist.most_common(most_common_count)
  return [word_tuple[0] for word_tuple in top_words]

def filter_most_common(train_texts, test_texts):
  train_file_path = get_common_words_texts_path('train')
  test_file_path = get_common_words_texts_path('test')

  text = ''
  top_words = []

  if os.path.isfile(train_file_path):
    with open(train_file_path, 'rb') as common_words_file:
      train_common_words = pickle.load(common_words_file)
      print('Train common words loaded from disk')
  else:
    text = train_texts.str.cat(sep=' ')
    top_words = get_most_common_words(text)
    train_common_words = train_texts.apply(lambda x: ' '.join([word for word in get_words(x) if word in top_words]))
    
    with open(train_file_path, 'wb') as common_words_file:
      pickle.dump(train_common_words, common_words_file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Train common words stored on disk')

  if os.path.isfile(test_file_path):
    with open(test_file_path, 'rb') as common_words_file:
      test_common_words = pickle.load(common_words_file)
      print('Test common words loaded from disk')
  else:
    if not text:
      text = train_texts.str.cat(sep=' ')
      top_words = get_most_common_words(text)

    test_common_words = test_texts.apply(lambda x: ' '.join([word for word in get_words(x) if word in top_words]))
    
    with open(test_file_path, 'wb') as common_words_file:
      pickle.dump(test_common_words, common_words_file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Test common words stored on disk')
    
  return train_common_words, test_common_words

def get_embedding_weights(word_vectors, word_index):
  weights_matrix = zeros((len(word_index) + 1, embedding_dims))

  count = 0
  for word, idx in word_index.items():
    if word in word_vectors:
      weights_matrix[idx] = word_vectors[word]
      count += 1
  print('Words found on word2vec: {}'.format(count))

  return weights_matrix 


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('type', choices=['training', 'crowdsourced_training'],
                      help='Select the type of dataset to use for training')
  parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                      help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
  parser.add_argument("--generate", '-g', action="store_true", default="False",
                      help="Use this argument to generate new data. By default it will load data from disk")
  parser.add_argument("--epochs", '-e', default="10",
                      help="Use this argument to set the number of epochs. Default: 10")
  parser.add_argument("--filters", '-f', default="64",
                      help="Use this argument to set the number of filters. Default: 64")
  parser.add_argument("--kernel", '-k', default="4",
                      help="Use this argument to set the size of kernels. Default: 4")
  parser.add_argument("--words", '-w', default="10000",
                      help="Use this argument to set the number of most common words to use. Default: 10000")
  args = parser.parse_args()
  
  global dataset_type
  dataset_type = args.type

  global sem_eval_path
  sem_eval_path = args.path

  global most_common_count
  most_common_count = int(args.words)

  epochs = int(args.epochs)
  filters = int(args.filters)
  kernel_size = int(args.kernel)
  hidden_dims = 250
  batch_size = 32 # default

  # 1. Load the texts
  X_train_texts_all, y_train, X_test_texts_all, y_test = load_texts()
  print('Texts loaded')
  print('Train texts shape: {}'.format(X_train_texts_all.shape))
  print('Test texts shape: {}'.format(X_test_texts_all.shape))

  X_train_texts, X_test_texts = filter_most_common(X_train_texts_all, X_test_texts_all)
  del X_train_texts_all
  del X_test_texts_all

  # 2. Encode texts as sequences
  tokenizer = load_tokenizer(X_train_texts)
  print('Word index size: {}'.format(len(tokenizer.word_index)))
  train_sequences = tokenizer.texts_to_sequences(X_train_texts)
  # print(set([word for seq in train_sequences for word in seq]))

  # # Check sequences length distribution
  # import numpy as np
  # data = np.array([len(seq) for seq in train_sequences])
  # mean = data.mean()
  # std = data.std()
  # max_val = data.max()
  # print('Mean: {}'.format(data.mean()))
  # print('Std: {}'.format(data.std()))
  # hist,bins=np.histogram(data, bins=[0.0, 2500.0, max_val])
  # print(hist)
  # print(bins)
  # return
  
  del X_train_texts
  print('Train sequences generated: {}'.format(len(train_sequences)))

  # 3. Pad sequenecs to have the same length
  print('Seq_len: {}'.format(seq_len))
  
  with tf.device('/cpu:0'):
    print('Padding training data on CPU...')
    X_train = pad_sequences(train_sequences, maxlen=seq_len, padding='post')
    del train_sequences
  print('Train sequences padded')

  test_sequences = tokenizer.texts_to_sequences(X_test_texts)
  del X_test_texts
  
  print('Test sequences generated: {}'.format(len(test_sequences)))
  test_seq_len = max([len(seq) for seq in test_sequences])
  
  with tf.device('/cpu:0'):
    print('Padding testing data on CPU...')
    X_test = pad_sequences(test_sequences, maxlen=seq_len, padding='post')
  del test_sequences
  print('Test sequences padded')

  print('-------------------------------------------------')

  model_path = get_model_path()

  # 4. Vocab size
  vocab_size = len(tokenizer.word_index) + 1
  print('Vocab size: {}'.format(vocab_size))

  # 5. Load word vectors
  word_vectors = load_word_vectors()
  print('Loaded word2vec')

  # 6. Create weights matrix
  weights_matrix = get_embedding_weights(word_vectors, tokenizer.word_index)
  print('weights matrix generated: {}'.format(weights_matrix.shape))
  
  # Remove word_vectors to free up memory
  del word_vectors

  # 7. Create Embeddings layer
  embedding_layer = Embedding(input_dim=vocab_size, 
                            output_dim=embedding_dims, 
                            weights=[weights_matrix],
                            input_length=seq_len,
                            trainable=False
                            )
  
  # 8. # Model definition
  model = Sequential()

  model.add(embedding_layer)

  # model.add(Conv1D(filters,
  #                 kernel_size,
  #                 activation='relu'))
  # model.add(MaxPooling1D(pool_size=2))

  model.add(Conv1D(filters,
                  kernel_size,
                  activation='relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling1D(pool_size=4))

  model.add(Conv1D(filters,
                  kernel_size,
                  activation='relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling1D(pool_size=4))

  model.add(Flatten())

  model.add(Dense(hidden_dims, activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(1, activation='sigmoid'))

  print(model.summary())

  print('Compiling model')
  model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

  # Model training
  print('Training model')
  model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(X_test, y_test))

  # Model Evaluation
  # scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
  # for idx, metric in enumerate(model.metrics_names):
  #   print('{}: {}'.format(metric, scores[idx]))
  
  # Save model to disk
  model.save(model_path)
  
if __name__ == "__main__":
    main()