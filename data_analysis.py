import argparse
import os
import logging
import sys
from data_loaders import TextsLoader
from keras.preprocessing.text import Tokenizer
import pickle

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                      help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
  parser.add_argument("--local", '-l', action='store_true', default="False",
                      help="Use this argument to work in local mode")
  args = parser.parse_args()

  sem_eval_path = args.path
  local = args.local

  crowdsourced = True if local is True else False

  logs_path = os.path.join(sem_eval_path, 'logs', '{}_log.log'.format(os.path.basename(__file__)))

  if local is True:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='\n%(message)s')
  else:
    logging.basicConfig(filename=logs_path, filemode='w', 
                    format='%(asctime)s : %(levelname)s : \n%(message)s', level=logging.INFO)

  use_mixed_datasets = False

  texts_loader = TextsLoader(sem_eval_path, crowdsourced, logs_path)

  if use_mixed_datasets:
    train_texts, y_train, val_texts, y_val = texts_loader.load_mixed()
  else:
    train_texts, y_train = texts_loader.load()
    val_texts, y_val = texts_loader.load(validation=True)
  
  logging.info('Train shape: {}'.format(train_texts.shape))
  logging.info('Number of biased samples: {}'.format(len(y_train[y_train == 1])))
  logging.info('Number of non-biased samples: {}'.format(len(y_train[y_train == 0])))

  logging.info('Validation shape: {}'.format(val_texts.shape))
  logging.info('Number of biased samples: {}'.format(len(y_val[y_val == 1])))
  logging.info('Number of non-biased samples: {}'.format(len(y_val[y_val == 0])))

  train_file_path = os.path.join(sem_eval_path, 'data', 'Tokenizers', 'data_analysis_train{}.pickle'.format('_mixed' if use_mixed_datasets else ''))
  if os.path.isfile(train_file_path):
    with open(train_file_path, 'rb') as tokenizer_file:
      tokenizer = pickle.load(tokenizer_file)
  else:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    with open(train_file_path, 'wb') as tokenizer_file:
      pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)
  train_words = list(tokenizer.word_index.keys())
  logging.info('train_words length: {}'.format(len(train_words)))
  logging.info('top train words: {}'.format(train_words[:100]))

  val_file_path = os.path.join(sem_eval_path, 'data', 'Tokenizers', 'data_analysis_val{}.pickle'.format('_mixed' if use_mixed_datasets else ''))
  if os.path.isfile(val_file_path):
    with open(val_file_path, 'rb') as tokenizer_file:
      val_tokenizer = pickle.load(tokenizer_file)
  else:
    val_tokenizer = Tokenizer()
    val_tokenizer.fit_on_texts(val_texts)
    with open(val_file_path, 'wb') as tokenizer_file:
      pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)
  val_words = list(val_tokenizer.word_index.keys())
  logging.info('val_words length: {}'.format(len(val_words)))
  logging.info('top val words: {}'.format(val_words[:100]))

  common_words = list(set(train_words) & set(val_words))
  logging.info('common_words length: {}'.format(len(common_words)))
  

if __name__ == "__main__":
    main()