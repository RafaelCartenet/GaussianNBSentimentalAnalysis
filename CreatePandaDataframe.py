import glob
import pandas as pd
import numpy as np

# PANDA DataFrame creation
#   output : creation of dataframe.pkl file

# Path Names
path ='data/txt_sentoken'
posfilenames = glob.glob(path + "/pos/*.txt")
negfilenames = glob.glob(path + "/neg/*.txt")

# Dataset structure
DFdict = {"review" : [],
          "sentiment" : []}

# Loading positive reviews
print "Loading Positive Reviews ..."
sentiment = "pos"
for filename in posfilenames:
    with open(filename) as f:
        review = " ".join(f.readlines())
        DFdict["review"].append(review)
        DFdict["sentiment"].append(sentiment)
print "DONE"

# Loading negative reviews
print "Loading Negative Reviews ..."
sentiment = "neg"
for filename in negfilenames:
    with open(filename) as f:
        review = " ".join(f.readlines())
        DFdict["review"].append(review)
        DFdict["sentiment"].append(sentiment)
print "DONE"

# Create Panda Dataframe
print "Creating Panda DataFrame ..."
NBreviews = len(DFdict["review"])
dataframe = pd.DataFrame(DFdict, index = range(0, NBreviews))
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
print "DONE"

print "Creating pkl file ..."
dataframe.to_pickle("data/dataframe.pkl")
print "dataframe.pkl created."
