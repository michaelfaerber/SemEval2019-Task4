from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Activation, Embedding, Flatten, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.engine.input_layer import Input
import argparse
from numpy import zeros
import clean_shuffle
from para2vec import ParagraphVectorModel, get_vector_label_mapping
from gensim.models.doc2vec import Doc2Vec
import numpy
import pickle
import os
import pandas as pd
import ground_truth_sqlite
import tensorflow as tf
from gensim.models import KeyedVectors


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sem_eval_path = ''
most_common_count = 100000
seq_len = 5000 # 2500 # Inferred from checking the sequences length distributions
embedding_dims = 300


def load_word_vectors():
  print('Loading word vectors...')
  filename = '{}/GoogleNews-vectors-negative300.bin'.format(sem_eval_path)
  model = KeyedVectors.load_word2vec_format(filename, binary=True)
  return model.wv

def load_texts():
    filename = os.path.join(sem_eval_path, 'data', 'IntegratedFiles', 'buzzfeed_training_withid.tsv')# 'crowdsourced_train_withid.tsv')
    df_location = os.path.join(sem_eval_path, 'data', 'Pickles', 'training_df.pickle')

    df = clean_shuffle.read_prepare_df(filename, file_path=df_location)

    ids_to_labels = ground_truth_sqlite.select_id_hyperpartisan_mappings(sem_eval_path, 'ground_truth_training') # 'ground_truth_crowdsourced_train'
    df['hyperpartisan'] = df.apply(lambda row: 1 if ids_to_labels[row['id']] == 'true' else 0, axis=1)

    df["text"] = df["title"] + ' ' + df["content"]

    return df['text'], df['hyperpartisan']

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    parser.add_argument("--epochs", '-e', default="10",
                        help="Use this argument to set the number of epochs. Default: 10")
    parser.add_argument("--filters", '-f', default="64",
                        help="Use this argument to set the number of filters. Default: 64")
    parser.add_argument("--kernel", '-k', default="4",
                        help="Use this argument to set the size of kernels. Default: 4")
    args = parser.parse_args()
    
    global sem_eval_path
    sem_eval_path = args.path

    # Hyperparameters
    filters = int(args.filters)
    kernel_size = int(args.kernel)
    epochs = int(args.epochs)
    hidden_dims = 250
    batch_size = 32 # default

    # Get data
    texts, y_train = load_texts()

    tokenizer = load_tokenizer(texts)

    train_sequences = tokenizer.texts_to_sequences(texts)
    del texts

    with tf.device('/cpu:0'):
        X_train = pad_sequences(train_sequences, maxlen=seq_len, padding='post')
        del train_sequences

    vocab_size = len(tokenizer.word_index) + 1
    print('Vocab size: {}'.format(vocab_size))

    # 5. Load word vectors
    word_vectors = load_word_vectors()
    weights_matrix = get_embedding_weights(word_vectors, tokenizer.word_index)
    
    # Remove word_vectors to free up memory
    del word_vectors

    # 7. Create Embeddings layer
    embedding_layer = Embedding(input_dim=vocab_size, 
                                output_dim=embedding_dims, 
                                weights=[weights_matrix],
                                input_length=seq_len,
                                trainable=False
                                )

    # Model definition
    model = Sequential()

    model.add(embedding_layer)

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

    model.add(GlobalMaxPooling1D())
    # model.add(Flatten())

    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2)

    conv_model_location = os.path.join(sem_eval_path, 'models', 'words_conv_model.h5')
    model.save(conv_model_location)

if __name__ == "__main__":
    main()