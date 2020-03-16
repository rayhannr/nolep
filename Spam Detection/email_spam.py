from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

dataset = pd.read_csv('spambase.data').as_matrix()
np.random.shuffle(dataset)

X = dataset[:, :48]
Y = dataset[:, -1]

X_train = X[:-100,]
Y_train = Y[:-100,]
X_test = X[-100:,]
Y_test = Y[-100:,]

model = MultinomialNB()
model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)