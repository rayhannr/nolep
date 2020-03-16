from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

dataset = pd.read_csv('spam.csv', encoding='ISO-8859-1')
dataset = dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
dataset.columns = ['labels', 'data']

dataset['b_labels'] = dataset['labels'].map({'ham': 0, 'spam': 1})
Y = dataset['b_labels'].as_matrix()

count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(dataset['data'])
X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

model = MultinomialNB()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

def visualize(label):
  words = ''
  for msg in dataset[dataset['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()
  
visualize('spam')
visualize('ham')

dataset['predictions'] = model.predict(X)

sneaky_spam = dataset[(dataset['predictions'] == 0) & (dataset['b_labels'] == 1)]['data']
  
not_actually_spam = dataset[(dataset['predictions'] == 1) & (dataset['b_labels'] == 0)]['data']