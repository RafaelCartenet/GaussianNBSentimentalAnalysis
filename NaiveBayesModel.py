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

def resultats(pred, real):
    TP = ((pred == "pos") & (real == "pos")).sum()
    FP = ((pred == "pos") & (real == "neg")).sum()
    FN = ((pred == "neg") & (real == "pos")).sum()
    TN = ((pred == "neg") & (real == "neg")).sum()
    recall = 100 * (TP / (float(TP) + FN))
    precision = 100 * (TP / (float(TP) + FP))
    F1score = 2*precision*recall / (precision + recall)
    return recall, precision, F1score

# DATA LOADING
print "Loading Dataframe ..."
datapath = "data/dataframe.pkl"
reviews, sentiments = data_loading(datapath)
ndata = reviews.size
print "DONE"

# DATA PRE-PROCESSING
print "Data Pre-Processing ..."
Words = ReviewsToWords(reviews)
BOWs = np.array(WordsToBOW(Words))
ndata_train = int(0.90*ndata)
ndata_test = ndata - ndata_train
BOWs_train, sentiments_train = BOWs[:ndata_train], sentiments[:ndata_train]
BOWs_test, sentiments_test = BOWs[:ndata_test], sentiments[:ndata_test]
print "DONE"

# MODEL TRAINING
print "Model Fitting ..."
model = GaussianNB()
model.fit(BOWs_train, sentiments_train)
print "DONE"
print

# PREDICTIONS
print "Predictions on Train dataset :"
y_pred = model.predict(BOWs_train)
recall, precision, F1score = resultats(y_pred, sentiments_train)
print "Recall    : %.3f" % recall, "%"
print "Precision : %.3f" % precision, "%"
print "F1score   : %.3f" % F1score, "%"
print

print "Predictions on Test dataset :"
y_pred = model.predict(BOWs_test)
recall, precision, F1score = resultats(y_pred, sentiments_test)
print "Recall    : %.3f" % recall, "%"
print "Precision : %.3f" % precision, "%"
print "F1score   : %.3f" % F1score, "%"
print
