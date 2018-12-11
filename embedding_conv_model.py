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

def current_file():
    return __file__.replace('.py', '')

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
      return  '{}/buzzfeed_{}.h5'.format(basepath, current_file()),
  elif dataset_type == 'crowdsourced_training':
      return  '{}/crowdsourced_{}.h5'.format(basepath, current_file())
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

  boundary = int(0.8*train_df['text'].shape[0])
  return train_df['text'][:boundary], train_df['hyperpartisan'][:boundary], train_df['text'][boundary:], train_df['hyperpartisan'][boundary:]

def load_test_texts():
  test_data_file = get_testing_tsv_path()
  test_dataframe_path = get_testing_dataframe_path()
  test_df = clean_shuffle.read_prepare_df(test_data_file, file_path=test_dataframe_path)

  test_df["text"] = test_df["title"] + ' ' + test_df["content"]

  return test_df['text'], test_df['hyperpartisan']

def load_texts():
  X_train, y_train, X_val, y_val = load_train_texts()
  X_test, y_test = load_test_texts()
  return X_train, y_train, X_val, y_val, X_test, y_test

def load_tokenizer(X_train_texts):
    num_words = 0 #most_common_count + 1
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

def filter_most_common(train_texts, val_texts, test_texts):
  text = train_texts.str.cat(sep=' ')
  top_words = get_most_common_words(text)

  train_common_words = train_texts.apply(lambda x: ' '.join([word for word in get_words(x) if word in top_words]))

  val_common_words = val_texts.apply(lambda x: ' '.join([word for word in get_words(x) if word in top_words]))

  test_common_words = test_texts.apply(lambda x: ' '.join([word for word in get_words(x) if word in top_words]))
    
  return train_common_words, val_common_words, test_common_words

def get_embedding_weights(word_vectors, word_index):
  weights_matrix = zeros((len(word_index) + 1, embedding_dims))

  for word, idx in word_index.items():
    # count += 1
    # if count == rows:
    #   break
    if word in word_vectors:
      weights_matrix[idx] = word_vectors[word]

  return weights_matrix 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['training', 'crowdsourced_training'],
                        help='Select the type of dataset to use for training')
    parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    parser.add_argument("--epochs", '-e', default="2",
                        help="Use this argument to set the number of epochs. Default: 2")
    parser.add_argument("--filters", '-f', default="64",
                        help="Use this argument to set the number of filters. Default: 64")
    parser.add_argument("--kernel", '-k', default="4",
                        help="Use this argument to set the size of kernels. Default: 4")
    parser.add_argument("--words", '-w', default="100000",
                        help="Use this argument to set the number of most common words to use. Default: 10000")
    args = parser.parse_args()
    
    batch_size = 32 # default
    hidden_dims = 250
    filters = int(args.filters)
    kernel_size = int(args.kernel)

    global dataset_type
    dataset_type = args.type

    global sem_eval_path
    sem_eval_path = args.path

    global most_common_count
    most_common_count = int(args.words)

    epochs = int(args.epochs)

    X_train_texts_all, y_train, X_val_texts_all, y_val, X_test_texts_all, y_test = load_texts()
    print('Texts loaded')
    print('Train texts shape: {}'.format(X_train_texts_all.shape))
    print('Train labels shape: {}'.format(y_train.shape))
    print('Val texts shape: {}'.format(X_val_texts_all.shape))
    print('Val labels shape: {}'.format(y_val.shape))
    print('Test texts shape: {}'.format(X_test_texts_all.shape))
    print('Test labels shape: {}'.format(y_test.shape))

    # X_train_texts = X_train_texts_all
    # X_val_texts = X_val_texts_all
    # X_test_texts = X_test_texts_all
    X_train_texts, X_val_texts, X_test_texts = filter_most_common(X_train_texts_all, X_val_texts_all, X_test_texts_all)
    del X_train_texts_all
    del X_val_texts_all
    del X_test_texts_all

    # tokenizer = load_tokenizer(X_train_texts)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_texts)
    print('Word index size: {}'.format(len(tokenizer.word_index)))
    print('Seq_len: {}'.format(seq_len))

    #Train set
    train_sequences = tokenizer.texts_to_sequences(X_train_texts)
    del X_train_texts
    print('Train sequences generated: {}'.format(len(train_sequences)))
    
    with tf.device('/cpu:0'):
        print('Padding training data on CPU...')
        X_train = pad_sequences(train_sequences, maxlen=seq_len, padding='post')
        del train_sequences
    print('Train sequences padded')

    #Validation set
    val_sequences = tokenizer.texts_to_sequences(X_val_texts)
    del X_val_texts
    print('Validation sequences generated: {}'.format(len(val_sequences)))
    
    with tf.device('/cpu:0'):
        print('Padding validation data on CPU...')
        X_val = pad_sequences(val_sequences, maxlen=seq_len, padding='post')
        del val_sequences
    print('Val sequences padded')

    # Test set
    test_sequences = tokenizer.texts_to_sequences(X_test_texts)
    del X_test_texts
    print('Test sequences generated: {}'.format(len(test_sequences)))
    
    with tf.device('/cpu:0'):
        print('Padding test data on CPU...')
        X_test = pad_sequences(test_sequences, maxlen=seq_len, padding='post')
    del test_sequences
    print('Test sequences padded')

    print('-------------------------------------------------')

    model_path = get_model_path()

    vocab_size = len(tokenizer.word_index) + 1
    print('Vocab size: {}'.format(vocab_size))

    # 8. # Model definition
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dims, 
                        input_length=seq_len
                                ))

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

    print('Training model')
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(X_val, y_val))

    scores = model.evaluate(X_test, y_test, batch_size=batch_size)
    for idx, metric in enumerate(model.metrics_names):
        print('{}: {}'.format(metric, scores[idx]))

    model.save(model_path)

if __name__ == "__main__":
    main()