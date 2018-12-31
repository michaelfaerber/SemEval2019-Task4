from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Activation, Embedding, Flatten, GlobalMaxPooling1D, LSTM
from keras import regularizers, callbacks, optimizers
from keras.models import load_model
import argparse
import os
import logging
from data_loaders import TextsLoader, TokenizerLoader, WordVectorsLoader, TextSequencesLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
conv_version = 5
lstm_version = 1
conv_lstm_version = 1

sem_eval_path = ''
seq_len = 2064 # 5000 # 2500 # Inferred from checking the sequences length distributions
embedding_mode = 0
crowdsourced = False
algorithm = 0

def load_embedding_layer(tokenizer):
    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    logging.info('Vocab size: {}'.format(vocab_size))

    # Load word vectors
    word_vectors_loader = WordVectorsLoader(sem_eval_path, crowdsourced, embedding_mode)
    word_vectors_loader.load()
    weights_matrix = word_vectors_loader.create_embedding_weights_matrix(tokenizer.word_index)
    
    return Embedding(input_dim=vocab_size, 
                                output_dim=weights_matrix.shape[1], 
                                weights=[weights_matrix],
                                input_length=seq_len,
                                trainable=False
                                )

def define_conv_model(tokenizer, filters=64, kernel_size=4, hidden_dims=256):
    model = Sequential()

    embedding_layer = load_embedding_layer(tokenizer)
    model.add(embedding_layer)

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

    # model.add(GlobalMaxPooling1D())
    model.add(Flatten())

    model.add(Dense(hidden_dims, activation='linear', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

def define_lstm_model(tokenizer, units=128):
    model = Sequential()

    embedding_layer = load_embedding_layer(tokenizer)
    model.add(embedding_layer)

    model.add(LSTM(units, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model

def define_conv_lstm_model(tokenizer, units=256, filters=64, kernel_size=4):
    model = Sequential()

    embedding_layer = load_embedding_layer(tokenizer)
    model.add(embedding_layer)

    model.add(Conv1D(filters,
                    kernel_size,
                    activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(filters,
                    kernel_size,
                    activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=2))

    model.add(LSTM(units, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))

    return model

def new_model_name():
    alg = ''
    version = 1
    if algorithm == 0:
        alg = 'conv'
        version = conv_version
    elif algorithm == 1:
        alg = 'conv_lstm'
        version = conv_lstm_version
    elif algorithm == 2:
        alg = 'lstm'
        version = lstm_version
    else:
        raise Exception('Unknown algorithm')
    return 'words_{}_model_w{}_v{}'.format(alg, embedding_mode, version)

def main():         
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    parser.add_argument("--crowdsourced", '-c', action='store_true', default="False",
                        help="Use this argument to work with the crowdsourced file")
    parser.add_argument("--model", '-m', default="",
                        help="Use this argument to continue training a stored model")
    parser.add_argument("--word_vectors", '-w', default="0",
                        help="Use this argument to set the word vectors to use: 0: Google's Word2vec, 1: GloVe, 2: Fasttext, 3: Custom pretrained word2vec. Default: 0")
    parser.add_argument("--algorithm", '-a', default="0",
                        help="Use this argument to set the algorithm to use: 0: CNN, 1: CNN + LSTM, 2: LSTM. Default: 0")
    args = parser.parse_args()
    
    global sem_eval_path
    sem_eval_path = args.path

    global embedding_mode
    embedding_mode = int(args.word_vectors)

    global algorithm
    algorithm = int(args.algorithm)

    global seq_len
    if algorithm == 0:
        seq_len = 5000
    elif algorithm == 1:
        seq_len = 2064
    elif algorithm == 2:
        seq_len = 500
    else:
        raise Exception('Unknown algorithm')

    model_name = args.model
    model_path = os.path.join(sem_eval_path, 'models')
    model_location = os.path.join(model_path, '{}.h5'.format(new_model_name()))
    model_weights_location = os.path.join(model_path, '{}.h5'.format(new_model_name()))

    logs_path = os.path.join(sem_eval_path, 'logs', '{}_log.log'.format(model_name if model_name else new_model_name()))
    logging.basicConfig(filename=logs_path, filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('model_location: {}'.format(model_location))
    
    global crowdsourced
    crowdsourced = args.crowdsourced

    batch_size = 32 # default

    # Get data
    train_texts, y_train, val_texts, y_val = TextsLoader(sem_eval_path, crowdsourced, logs_path).load_mixed()
    logging.info('Train shape: {}'.format(train_texts.shape))
    logging.info('Validation shape: {}'.format(val_texts.shape))
    
    tokenizer = TokenizerLoader(train_texts, sem_eval_path, logs_path).load()
    sequences_loader = TextSequencesLoader(tokenizer, seq_len)
    X_train = sequences_loader.load(train_texts)
    X_val = sequences_loader.load(val_texts)

    if model_name:
        model_path = os.path.join(sem_eval_path, 'models', "{}.h5".format(model_name))
        model = load_model(model_path)
    elif algorithm == 0:
        model = define_conv_model(tokenizer)
    elif algorithm == 1:
        model = define_conv_lstm_model(tokenizer)
    elif algorithm == 2:
        model = define_lstm_model(tokenizer)
    else:
        raise Exception('Unknown algorithm')

    logging.info(model.summary())

    # Implement Early Stopping
    early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1)
                            #   restore_best_weights=True)
    save_best_model = callbacks.ModelCheckpoint(model_weights_location, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])

    model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=50,
                verbose=2,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping_callback, save_best_model])
    
    #reload best weights
    model.load_weights(model_weights_location)

    logging.info('Model trained. Storing model on disk.')
    model.save(model_location)
    logging.info('Model stored on disk.')

if __name__ == "__main__":
    main()