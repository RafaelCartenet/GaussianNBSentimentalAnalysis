import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

# Naive Bayes Model for Sentimental Analysis
#   loads imbd dataframe
#   train Gaussian Nayve Bayes model
#   predicts

# FUNCTIONS
def data_loading(filename):
    dataframe = pd.read_pickle(filename)
    x_data = np.array(dataframe.ix[:,0])
    y_label = np.array(dataframe.ix[:,1])
    return x_data, y_label

uselesswords = set(stopwords.words("english"))
def ReviewsToWords(reviews):
    Words = []
    for i in range(len(reviews)):
        review = bs(reviews[i], "lxml").get_text()
        words = re.sub("[^a-zA-Z]", " ", review).lower().split()
        words = [word for word in words if word not in uselesswords]
        words = " ".join(words)
        Words.append(words)
    return Words

def WordsToBOW(words):
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000)
    BOWs = vectorizer.fit_transform(words)
    return BOWs.toarray()

# DATA LOADING
print "Loading Dataframe ..."
datapath = "data/dataframe.pkl"
reviews, sentiments = data_loading(datapath)
print "DONE"

# DATA PRE-PROCESSING
print "Data Pre-Processing ..."
Words = ReviewsToWords(reviews)
BOWs = np.array(WordsToBOW(Words))
print "DONE"

# MODEL TRAINING
print "Model Fitting ..."
model = GaussianNB()
model.fit(BOWs, sentiments)
print "DONE"

# PREDICTIONS
print "Predictions ..."
y_pred = model.predict(BOWs)
N = len(y_pred)
print "DONE"

# RESULTS
nbfails = (y_pred != sentiments).sum()
accuracy = 100*(1. - float(nbfails)/N)
print "Nb Fails : ", nbfails
print "Accuracy : %.2f" % accuracy, "%"
