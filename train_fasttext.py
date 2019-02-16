import argparse
from data_loaders import TextsLoader
from gensim.models import FastText
import nltk
import os

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                      help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
  parser.add_argument("--crowdsourced", '-c', action='store_true', default="False",
                        help="Use this argument to work with the crowdsourced file")
  args = parser.parse_args()
  
  sem_eval_path = args.path
  crowdsourced = args.crowdsourced

  X_train = TextsLoader(sem_eval_path, crowdsourced).load_mixed()[0]
  wpt = nltk.WordPunctTokenizer()
  tokenized_corpus = [wpt.tokenize(document) for document in X_train.values]
  
  model = FastText(tokenized_corpus, size=100, window=3, min_count=1, iter=10)

  existent_word = "trump"
  print(existent_word in model.wv.vocab)

  dataset_name = 'crowdsourced_' if crowdsourced is True else ''
  path = os.path.join(sem_eval_path, 'models', '{}custom_fasttext.bin'.format(dataset_name))
  model.save(path)

  new_model = FastText.load(path)
  print(new_model)

if __name__ == "__main__":
    main()