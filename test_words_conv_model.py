from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import argparse
import clean_shuffle
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

def load_texts():
    filename = os.path.join(sem_eval_path, 'data', 'IntegratedFiles', 'buzzfeed_validation_withid.tsv')# 'crowdsourced_train_withid.tsv')
    df_location = os.path.join(sem_eval_path, 'data', 'Pickles', 'validation_df.pickle')

    df = clean_shuffle.read_prepare_df(filename, file_path=df_location)

    ids_to_labels = ground_truth_sqlite.select_id_hyperpartisan_mappings(sem_eval_path, 'ground_truth_validation') # 'ground_truth_crowdsourced_train'
    df['hyperpartisan'] = df.apply(lambda row: 1 if ids_to_labels[row['id']] == 'true' else 0, axis=1)

    df["text"] = df["title"] + ' ' + df["content"]

    return df['text'], df['hyperpartisan']

def load_train_tokenizer():
    num_words = most_common_count + 1
    file_path = os.path.join(sem_eval_path, 'data', 'Tokenizers', 'buzzfeed_trained_{}_tokenizer.pickle'.format(num_words))

    with open(file_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

    return tokenizer 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file_name',
                        help='Set the file name of the trained model, ex: words_conv_model')
    parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    args = parser.parse_args()
    
    global sem_eval_path
    sem_eval_path = args.path
    model_file_name = args.model_file_name
    batch_size = 32 # default

    # Get data
    texts, y_validation = load_texts()

    tokenizer = load_train_tokenizer()

    validation_sequences = tokenizer.texts_to_sequences(texts)
    del texts

    with tf.device('/cpu:0'):
        X_validation = pad_sequences(validation_sequences, maxlen=seq_len, padding='post')
        del validation_sequences
    
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocab size: {}'.format(vocab_size))
    
    print('Loading trained model...')
    model_path = os.path.join(sem_eval_path, 'models', "{}.h5".format(model_file_name))
    model = load_model(model_path)

    scores = model.evaluate(X_validation, y_validation, batch_size=batch_size, verbose=1)
    for idx, metric in enumerate(model.metrics_names):
      print('{}: {}'.format(metric, scores[idx]))


if __name__ == "__main__":
    main()