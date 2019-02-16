import argparse
import os
import clean_shuffle
import multiprocessing
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import logging
import pickle

cores = multiprocessing.cpu_count()
sem_eval_path = '/vol3/SemEval2019Task4/' #'/home/agon/Files/SemEval'
files_path = '/home/agon/Files/SemEval/jsi-news-asTxt'

def _parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sem_eval_path",'-s', default="/vol3/SemEval2019Task4/",
                      help="Use this argument to change the SemEval directory path (the default path is: '/vol3/SemEval2019Task4/')")
  parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval/jsi-news-asTxt",
                      help="Use this argument to change the files directory path (the default path is: '/home/agon/Files/SemEval/jsi-news-asTxt')")
  args = parser.parse_args()
  
  global files_path
  files_path = args.path
  global sem_eval_path
  sem_eval_path = args.sem_eval_path
  logging.basicConfig(filename='{}/logs/info_log.log'.format(sem_eval_path), filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def _read_text_files():
  files_count = len([name for name in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, name))])
  logging.info('Reading text files. Number of files: {}'.format(files_count))
  counter = 1
  sentences = []
  for filename in os.listdir(files_path):
    full_name = os.path.join(files_path, filename)
    article = open(full_name).read()
    article_sentences = _preprocess_texts([article])
    sentences += article_sentences
    if counter % 10000 == 0:
      logging.info('{} files done from {}'.format(counter, files_count))
    counter += 1
  return sentences

def _preprocess_texts(texts):
  sentences = []
  for article in texts:
    sent_text = sent_tokenize(article)
    for sentence in sent_text:
      sentence = clean_shuffle.clean_text(sentence)
      words_array = word_tokenize(sentence)
      sentences.append(words_array)
  return sentences

def _load_sentences():
  sentences_file_path = os.path.join(sem_eval_path, 'models', 'news_texts', "sentences.txt")
  if os.path.isfile(sentences_file_path):
    logging.info('Loading saved sentences...')
    with open(sentences_file_path, "rb") as fp:
      sentences = pickle.load(fp)
  else:
    logging.info('Generating new sentences...')
    sentences = _read_text_files()
    with open(sentences_file_path, "wb") as fp:
      pickle.dump(sentences, fp)
  return sentences  

def main():
  _parse_arguments()

  sentences = _load_sentences()

  # Create CBOW model 
  logging.info('Creating CBOW word2vec model...')
  model1 = gensim.models.Word2Vec(sentences, min_count = 1, size = 100, window = 5, workers=cores) 
  model1_path = os.path.join(sem_eval_path, 'models', 'news_texts', 'news_words2vec_cbow.bin')
  model1.save(model1_path)

  # # Create Skip Gram model 
  # logging.info('Creating Skip Gram word2vec model...')
  # model2 = gensim.models.Word2Vec(sentences, min_count = 1, size = 100, window = 5, sg = 1, workers=cores)
  # model2_path = os.path.join(sem_eval_path, 'models', 'news_texts', 'news_words2vec_sg.bin')
  # model2.save(model2_path)

  return

if __name__ == "__main__":
    main()