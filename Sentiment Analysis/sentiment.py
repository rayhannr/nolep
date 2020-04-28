import nltk
nltk.download('wordnet')
import re
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()

#stopwords = set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
#the file contains more positive reviews. we execute this code to make it balanced with the negative
positive_reviews = positive_reviews[:len(negative_reviews)]

def tokenize(s):
    s = s.lower()
    s = re.sub('[^a-zA-Z]', ' ', s)
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] #just like stemming
    tokens = [t for t in tokens if t not in set(stopwords.words('english'))]
    return tokens

word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = tokenize(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
for review in negative_reviews:
    tokens = tokenize(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x /= x.sum()
    x[-1] = label
    return x

N = len(positive_tokenized) - len(negative_tokenized)

data = np.zeros((2000, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1
    
np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1]

X_train = X[:-100,]
y_train = y[:-100,]
X_test = X[-100:,]
y_test = y[-100:,]

model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

prediction = model.predict(X_test)

threshold = 0.5
words = []
sentiment = []
for word in word_index_map:
    index = word_index_map[word]
    weight = model.coef_[0][index]
    words.append(word)
    sentiment.append(weight)

words = np.array(words)
sentiment = np.array(sentiment)
sentiment_weight = np.stack((words, sentiment)).T