import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import gc
import gensim
import nltk
import string
#nltk.download()

import multiprocessing as mp

from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn import ensemble
from sklearn import linear_model
from sklearn import pipeline
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import text as ftext
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sys import getsizeof
from vowpalwabbit import pyvw
from gensim.models import TfidfModel
from gensim import corpora
from gensim.parsing import preprocessing as prep
from collections import defaultdict
from sklearn.utils import shuffle

TRAIN_SIZE = 3600000
RND_STATE = 1234

# будем хранить в памяти только нужный в данный момент кусок датасета
mask_train, mask_test = train_test_split(
    np.arange(TRAIN_SIZE),
    test_size=0.12, # Получаем размер теста почти равный размеру private.
    shuffle=True,
    random_state=RND_STATE
)

def get_df(mask):
    texts_train = []
    with open('../input/x_train.txt') as f:
        texts_train = f.read().split('\n')
        texts_train.pop();

    df_train = pd.DataFrame()
    df_train['text'] = texts_train

    labels = pd.read_csv('../input/y_train.csv').drop('Id', axis=1)

    print('reading finished')

    df_train = pd.concat([df_train, labels], axis=1)

    df_train = df_train.iloc[mask]

    labels = df_train['Probability']
    df_train.drop('Probability', axis=1, inplace=True)

    return df_train['text'], labels

texts, labels = get_df(mask_train)
print('after get_df: collected {}'.format(gc.collect()))

class TokenizerNltk(object):
    def __call__(self, doc):
        striped = prep.strip_punctuation(doc)
        striped = prep.strip_tags(striped)
        striped = prep.strip_multiple_whitespaces(striped).lower()
        return striped

def prepare_texts(texts):
    tokenizer = TokenizerNltk()
    return Parallel(n_jobs=-1)(
        delayed(tokenizer)(t) for t in texts
    )

# Тут просто убираем пунктуацию, тэги <html>, <...>, убираем лишние пробелы и делаем слова нижним регистром.
texts = prepare_texts(texts)
print('after prepare_texts: collected {}'.format(gc.collect()))

class FilterRareWords(object):
    def __init__(self):
        self.cv = defaultdict(int)
    def fit(self, texts):
        for text in texts:
            for word in text.split():
                self.cv[word] += 1
    def __call__(self, text):
        return ' '.join([self.filter_word(word) for word in text.split()])
    def filter_word(self, word):
        return '' if self.cv[word] < 2 else word

# Зачем?
# В Vowpal Wabbit используется hashing trick максимальный размер числа определяется параметром 'b' и равен 2^b. Максимальное b=29, которое влезает на кернелах.
# Так как мы используем ngram=3 и skips=1, то каждое слово, которое встречалось один раз даст 8 разных хэшей, которые также встречались один раз.
# Таких слов примерно 1/3 от общего количества.
filter_words = FilterRareWords()
filter_words.fit(texts)

def filter_words_parallel(texts):
    return [filter_words(t) for t in texts]

texts = filter_words_parallel(texts)
print('after filter rare: collected {}'.format(gc.collect()))

# Держим готовый текст на диске, чтоб не занимать лишнее место
with open('corpus', 'w') as f:
    f.write('\n'.join(texts))

del texts
print('texts loaded to file: ', gc.collect())

def make_vw_feature_line(label, text):
    return '{} |text {}'.format(label, text)

def make_vw_corpus(file_name, labels):
    for text, label in zip(open(file_name), labels):
        yield make_vw_feature_line('1' if label == 1.0 else '-1', text)

def make_vw_corpus_file(file_name, labels):
    vw_corpus = [x.replace('\n', '') for x in make_vw_corpus(file_name, labels)]
    with open(file_name, 'w') as f:
        f.write('\n'.join(vw_corpus)) 
        
def make_corpus_it(file_name, limit):
    for feature, _ in zip(open(file_name), range(limit)):
        yield feature

# Приводим тексты на диске в формат VW        
make_vw_corpus_file('corpus', labels)
limit = len(labels)

print('loading X_test df')
X_test_df, y_test = get_df(mask_test)

X_test_df = prepare_texts(X_test_df)
print('after X_test prepare: ', gc.collect())

X_test_df = filter_words_parallel(X_test_df)

with open('corpus_test', 'w') as f:
    f.write('\n'.join(X_test_df))

del X_test_df
print('texts loaded to file: ', gc.collect())

# Тут самое интересное.
# с loss_function, quiet, link, random_seed все понятно.
# b -- ищем максимальный, при котором кернел не упадет (чем больше тем лучше)
# ngram и skips перебираем руками.
# l1, l2, learning_rate, passes перебираем с помощью vw-hyperopt.
# Примерная строчка запуска: 
# vw-hyperopt.py --train train_corpus --holdout holdout_corpus --outer_loss_function roc-auc --plot --max_evals 100 
#                --vw_space "--passes=1..5 -l=0.1..1.5 --l2=1e-13..1e-8 --l1=1e-13..1e-8 -b=29 --link=logistic --loss_function=logistic --skips=1 --ngram=3"
vw = pyvw.vw(
    quiet=True,
    loss_function='logistic',
    link='logistic',
    b=29,
    ngram=3,
    skips=1,
    l1=4.41234725256e-09,
    l2=4.07463874104e-11,
    random_seed=RND_STATE,
    learning_rate=0.75057834225,
    # ftrl=True
)

def get_pred(feature):
    ex = vw.example(feature)
    pred = vw.predict(ex)
    ex.finish()
    return pred

for fit_iter in range(5):
    for num, feature in enumerate(make_corpus_it('corpus', limit)):
        ex = vw.example(feature)
        vw.learn(ex)
        ex.finish()
        
    print('iter num {} done'.format(fit_iter))

pred = np.array([get_pred(x) for x in make_vw_corpus('corpus_test', y_test)])
print('auc score is {}'.format(roc_auc_score(y_test, pred)))
    
print('sumbit:')
def get_test_df():
    texts_test = []
    with open('../input/x_test.txt') as f:
        texts_test = f.read().split('\n')
        texts_test.pop();

    df_test = pd.DataFrame()
    df_test['text'] = texts_test

    print('test reading finished')

    return df_test['text']

test = get_test_df()

test = prepare_texts(test)
print('after test prepare: ', gc.collect())

test = filter_words_parallel(test)

with open('corpus', 'w') as f:
    f.write('\n'.join(test))

print('texts loaded to file: ', gc.collect())

pred = np.array([get_pred(x) for x in make_vw_corpus('corpus', [1] * len(test))])

example = pd.read_csv('../input/random_prediction.csv')
example['Probability'] = pred.astype(np.float64)
example.to_csv('submit_calibrated.csv', index=False)