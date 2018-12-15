import nltk
import string
import os
import pickle

class MostCommonWordsService:
  def __init__(self, count, sem_eval_path, regenerate=False, texts=''):
    self.count = count
    self.file_path = os.path.join(sem_eval_path, 'data', 'CommonWords', 'buzzfeed_train_{}_common_words.pickle'.format(count))
    self.regenerate = regenerate
    self.texts = texts

  def call(self):
    if os.path.isfile(self.file_path) and not self.regenerate:
      with open(self.file_path, 'rb') as common_words_file:
        most_common_words = pickle.load(common_words_file)
        print('Train common words loaded from disk')
    else:
      most_common_words = self._generate_texts_with_common_words()
    return most_common_words

  def _generate_texts_with_common_words(self):
    top_words = self._generate_most_common_words()
    most_common_words = self.texts.apply(lambda x: ' '.join([word for word in self._get_words(x) if word in top_words]))
    
    with open(self.file_path, 'wb') as common_words_file:
      pickle.dump(most_common_words, common_words_file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Train common words stored on disk: {}'.format(self.file_path))
    
    return most_common_words

  def _generate_most_common_words(self):
    entire_text = self.texts.str.cat(sep=' ')
    words = self._get_words(entire_text)
    fdist = nltk.probability.FreqDist(words)
    top_words = fdist.most_common(self.count)
    return [word_tuple[0] for word_tuple in top_words]

  def _get_words(self, text):
    tokens = nltk.tokenize.word_tokenize(text)
    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # remove 1-character words
    words = [word for word in words if len(word) > 1]
    return words