import pandas as pd
import numpy as np

import re

from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# Sentiment Analysis

# FUNCTIONS
def data_loading(filename):
    dataframe = pd.read_pickle(filename)
    x_data = np.array(dataframe.ix[:,0])
    y_label = np.array(dataframe.ix[:,1])
    return x_data, y_label

Stopwords = set(stopwords.words("english"))
def ReviewsToWords(reviews):
    Words = []
    for i in range(len(reviews)):
        review = bs(reviews[i], "lxml").get_text()
        words = re.sub("[^a-zA-Z]", " ", review).lower().split()
        words = [word for word in words if word not in Stopwords]
        words = " ".join(words)
        Words.append(words)
    return Words

def BOWvectorizer(words):
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 3000)
    return vectorizer.fit(words)

def train_nb(x, y, model = MultinomialNB):
    model = model()
    model.fit(x, y)
    return model

def classify_nb(model, x):
    return model.predict(x)

# DATA LOADING
print "Loading Dataframe ..."
datapath = "data/dataframe.pkl"
reviews, y = data_loading(datapath)
print "DONE"

# DATA SPLITTING
ndata = reviews.size
ndata_train = int(0.80*ndata)
ndata_test = ndata - ndata_train
reviews_train, y_train = reviews[:ndata_train], y[:ndata_train]
reviews_test, y_test = reviews[-ndata_test:], y[-ndata_test:]

# DATA PRE-PROCESSING
print "Data Pre-Processing ..."
# Transforming Reviews into lists of words
Words_train = ReviewsToWords(reviews_train)
Words_test = ReviewsToWords(reviews_test)

# Creating a Bag of Words vectorizer out of training data
Vectorizer = BOWvectorizer(Words_train)

# Vectorizing training and testing data
x_train = Vectorizer.transform(Words_train).toarray()
x_test = Vectorizer.transform(Words_test).toarray()
print "DONE"

# MODEL TRAINING
print "Model Fitting ..."
model = train_nb(x_train, y_train, MultinomialNB)
print "DONE"
print

# PREDICTIONS
print "Predictions on Train dataset :"
y_pred = classify_nb(model, x_train)
print classification_report(y_train, y_pred)

print "Predictions on Test dataset :"
y_pred = classify_nb(model, x_test)
print classification_report(y_test, y_pred)
