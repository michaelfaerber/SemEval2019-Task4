from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Activation, Embedding, Flatten, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers, callbacks, optimizers
from keras.models import load_model
import argparse
from numpy import zeros
import clean_shuffle
import numpy
import pickle
import os
import pandas as pd
import ground_truth_sqlite
import tensorflow as tf
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
version = 3

sem_eval_path = ''
most_common_count = 100000
seq_len = 5000 # 2500 # Inferred from checking the sequences length distributions
embedding_dims = 300
train_val_boundary = 0.8


def load_word_vectors():
  print('Loading word vectors...')
  filename = '{}/GoogleNews-vectors-negative300.bin'.format(sem_eval_path)
  model = KeyedVectors.load_word2vec_format(filename, binary=True)
  return model.wv

def load_texts(crowdsourced=False, split=True, validation=False):
    name = 'validation' if validation else 'training'
    tsv_name = 'crowdsourced_train' if crowdsourced is True else 'buzzfeed_{}'.format(name)
    table_name = 'crowdsourced_train' if crowdsourced is True else name
    df_name = 'crowdsourced_train_df' if crowdsourced is True else '{}_df'.format(name)

    filename = os.path.join(sem_eval_path, 'data', 'IntegratedFiles', '{}_withid.tsv'.format(tsv_name))
    df_location = os.path.join(sem_eval_path, 'data', 'Pickles', '{}.pickle'.format(df_name))

    df = clean_shuffle.read_prepare_df(filename, file_path=df_location)

    ids_to_labels = ground_truth_sqlite.select_id_hyperpartisan_mappings(sem_eval_path, 'ground_truth_{}'.format(table_name))
    df['hyperpartisan'] = df.apply(lambda row: 1 if ids_to_labels[row['id']] == 'true' else 0, axis=1)

    df["text"] = df["title"] + ' ' + df["content"]

    if split:
        boundary = int(train_val_boundary * df['text'].shape[0])
        return df['text'][:boundary], df['hyperpartisan'][:boundary], df['text'][boundary:], df['hyperpartisan'][boundary:]
    else:
        return df

def load_merged_texts(crowdsourced):
    train_df = load_texts(crowdsourced=crowdsourced, split=False, validation=False)
    val_df = load_texts(crowdsourced=crowdsourced, split=False, validation=True)

    base_path = os.path.join(sem_eval_path, 'data', 'Pickles')
    train_df_path = os.path.join(base_path, 'mixed_training_df.pickle')
    val_df_path = os.path.join(base_path, 'mixed_validation_df.pickle')

    if os.path.isfile(train_df_path) and os.path.isfile(val_df_path):
        print('Loading mixed dataframes from disk')
        train_df = pd.read_pickle(train_df_path)
        val_df = pd.read_pickle(val_df_path)
    else:
        print('Creating new mixed dataframes')

        # Append
        df = train_df.append(val_df, ignore_index=True)
        print('Appended shape: {}'.format(df.shape))
        print(df[:2])
        
        # Shuffle
        df = df.sample(frac=1).reset_index(drop=True)
        print(df[:2])

        # Split train/test
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['hyperpartisan'])

    print('Shapes after splitting:')
    print(train_df.shape)
    print(val_df.shape)

    return train_df['text'], train_df['hyperpartisan'], val_df['text'], val_df['hyperpartisan']

def load_tokenizer(texts):
    num_words = most_common_count + 1
    file_path = os.path.join(sem_eval_path, 'data', 'Tokenizers', 'buzzfeed_trained_{}_tokenizer.pickle'.format(num_words))

    if os.path.isfile(file_path):
        with open(file_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        print('Tokenizer loaded from disk')
    else:
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(texts)

        with open(file_path, 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)

        print('Tokenizer fit on texts and stored on disk')
    return tokenizer

def get_embedding_weights(word_vectors, word_index):
  weights_matrix = zeros((len(word_index) + 1, embedding_dims))

  count = 0
  for word, idx in word_index.items():
    if word in word_vectors:
      weights_matrix[idx] = word_vectors[word]
      count += 1
  print('Words found on word2vec: {}'.format(count))

  return weights_matrix 

def define_model(tokenizer, filters, kernel_size, hidden_dims):
    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocab size: {}'.format(vocab_size))

    # Load word vectors
    word_vectors = load_word_vectors()
    weights_matrix = get_embedding_weights(word_vectors, tokenizer.word_index)
    
    # Remove word_vectors to free up memory
    del word_vectors

    # Model definition
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, 
                                output_dim=embedding_dims, 
                                weights=[weights_matrix],
                                input_length=seq_len,
                                trainable=False
                                ))

    model.add(Conv1D(filters,
                    kernel_size,
                    activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(filters,
                    kernel_size,
                    activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=4))

    model.add(GlobalMaxPooling1D())
    # model.add(Flatten())

    model.add(Dense(hidden_dims, activation='linear', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    parser.add_argument("--epochs", '-e', default="50",
                        help="Use this argument to set the number of epochs. Default: 50")
    parser.add_argument("--filters", '-f', default="64",
                        help="Use this argument to set the number of filters. Default: 64")
    parser.add_argument("--kernel", '-k', default="4",
                        help="Use this argument to set the size of kernels. Default: 4")
    parser.add_argument("--crowdsourced", '-c', action='store_true', default="False",
                        help="Use this argument to work with the crowdsourced file")
    parser.add_argument("--model", '-m', default="",
                        help="Use this argument to continue training a model")
    args = parser.parse_args()
    
    global sem_eval_path
    sem_eval_path = args.path

    # Hyperparameters
    filters = int(args.filters)
    kernel_size = int(args.kernel)
    epochs = int(args.epochs)
    hidden_dims = 64
    batch_size = 32 # default
    model_name = args.model

    # Get data
    train_texts, y_train, val_texts, y_val = load_merged_texts(args.crowdsourced)
    print('Train shape: {}'.format(train_texts.shape))
    print('Validation shape: {}'.format(val_texts.shape))
    
    tokenizer = load_tokenizer(train_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    del train_texts
    with tf.device('/cpu:0'):
        X_train = pad_sequences(train_sequences, maxlen=seq_len, padding='post')
        del train_sequences

    val_sequences = tokenizer.texts_to_sequences(val_texts)
    del val_texts
    with tf.device('/cpu:0'):
        X_val = pad_sequences(val_sequences, maxlen=seq_len, padding='post')
        del val_sequences

    if model_name:
       model_path = os.path.join(sem_eval_path, 'models', "{}.h5".format(model_name))
       model = load_model(model_path)
    else:
       model = define_model(tokenizer, filters, kernel_size, hidden_dims)

    print(model.summary())

    # Implement Early Stopping
    early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=1,
                              restore_best_weights=True)
    print('Min delta: 0')
    
    adam = optimizers.Adam(lr=0.0005)
    model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])

    model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping_callback])

    conv_model_location = os.path.join(sem_eval_path, 'models', 'words_conv_model_{}.h5'.format(version))
    model.save(conv_model_location)

if __name__ == "__main__":
    main()