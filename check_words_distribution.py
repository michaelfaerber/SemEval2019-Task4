import clean_shuffle
from para2vec import ParagraphVectorModel, get_vector_label_mapping
import argparse
from gensim.models.doc2vec import Doc2Vec
from keras.preprocessing.text import Tokenizer


dataset_type = 'training'
sem_eval_path = ''

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
  return X_train, X_test

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                      help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
  args = parser.parse_args()

  global sem_eval_path
  sem_eval_path = args.path
  
  X_train, X_test = load_texts()

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_train)
  train_words = [word for word in tokenizer.word_counts]
  print('Total train words: {}'.format(len(train_words)))

  test_tokenizer = Tokenizer()
  test_tokenizer.fit_on_texts(X_test)
  test_words = [word for word in test_tokenizer.word_counts]
  print('Total test words: {}'.format(len(test_words)))

  intersection = set(train_words) & set(test_words)

  print('Total common words: {}'.format(len(intersection)))

if __name__ == "__main__":
    main()