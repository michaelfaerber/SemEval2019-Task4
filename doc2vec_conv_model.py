from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Activation, Embedding, Flatten
from gensim.models import KeyedVectors
import clean_shuffle
import argparse
from numpy import zeros

sem_eval_path = ''
dataset_type = ''
embedding_dims = 300
train_test_boundary = 0.8

def load_word_vectors():
  print('Loading word vectors...')
  filename = '{}/GoogleNews-vectors-negative300.bin'.format(sem_eval_path)
  model = KeyedVectors.load_word2vec_format(filename, binary=True)
  return model.wv

def get_tsv_path():
    basepath = '{}/data/IntegratedFiles'.format(sem_eval_path)
    if dataset_type == 'training':
        return  '{}/buzzfeed_training.tsv'.format(basepath)
    elif dataset_type == 'crowdsourced_training':
        return  '{}/crowdsourced_train.tsv'.format(basepath)
    return 'Dummy: not yet available'

def load_texts():
    data_file = get_tsv_path()
    df = clean_shuffle.read_prepare_df(data_file)

    df["text"] = df["title"] + ' ' + df["content"]

    boundary = int(train_test_boundary * df['text'].size)
    return df['text'][:boundary], df['hyperpartisan'][:boundary], df['text'][boundary:], df['hyperpartisan'][boundary:]

def get_weights_matrix(word_vectors, word_vocab):
  weights_matrix = zeros((len(word_vocab) + 1, embedding_dims))

  for word, idx in word_vocab.items():
    if word in word_vectors:
      weights_matrix[idx] = word_vectors[word]

  return weights_matrix 


def main():
  batch_size = 32 # default
  filters = 250
  kernel_size = 4
  hidden_dims = 250
  epochs = 10

  parser = argparse.ArgumentParser()
  parser.add_argument('type', choices=['training', 'crowdsourced_training'],
                      help='Select the type of dataset to use for training')
  parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                      help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
  parser.add_argument("--generate", '-g', action="store_true", default="False",
                      help="Use this argument to generate new data. By default it will use the loaded data")
  parser.add_argument("--epochs", '-e', default="10",
                      help="Use this argument to set the number of epochs. Default: 10")
  parser.add_argument("--filters", '-f', default="250",
                      help="Use this argument to set the number of filters. Default: 100")
  parser.add_argument("--kernel", '-k', default="4",
                      help="Use this argument to set the size of kernels. Default: 4")
  args = parser.parse_args()
  
  global dataset_type
  dataset_type = args.type

  global sem_eval_path
  sem_eval_path = args.path

  # 1. Load the texts
  X_train_texts, y_train, X_test_texts, y_test = load_texts()

  # 2. Encode texts as sequences
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_train_texts)
  train_sequences = tokenizer.texts_to_sequences(X_train_texts)

  # 3. Pad sequenecs to have the same length
  seq_len = max([len(seq) for seq in train_sequences])
  X_train = pad_sequences(train_sequences, maxlen=seq_len, padding='post')

  test_sequences = tokenizer.texts_to_sequences(X_test_texts)
  X_test = pad_sequences(test_sequences, maxlen=seq_len, padding='post')

  # 4. Vocab size
  vocab_size = len(tokenizer.word_index) + 1

  # 5. Load word vectors
  word_vectors = load_word_vectors()

  # 6. Create weights matrix
  weights_matrix = get_weights_matrix(word_vectors, tokenizer.word_index)
  
  # Remove word_vectors to free up memory
  del word_vectors
  # print([weights_matrix].shape)

  # 7. Create Embeddings layer
  embedding_layer = Embedding(input_dim=vocab_size, 
                              output_dim=embedding_dims, 
                              weights=[weights_matrix],
                              input_length=seq_len
                              )
  
  # 8. # Model definition
  model = Sequential()

  model.add(embedding_layer)

  model.add(Conv1D(filters,
                  kernel_size,
                  activation='relu'))
  model.add(MaxPooling1D(pool_size=2))

  model.add(Flatten())

  model.add(Dense(hidden_dims, activation='relu'))
  model.add(Dropout(0.2))

  model.add(Dense(1, activation='sigmoid'))

  # print(model.summary())

  model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

  model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(X_test, y_test))

if __name__ == "__main__":
    main()