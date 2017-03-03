import numpy as np
import nltk

stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

def remove_stopwords_and_stem(words):
    words = [token for token in words if (len(token)>2 and (token not in stpwds))]
    return [stemmer.stem(token) for token in words]

def random_sample(df, p = 0.05,seed=42):
    '''
    Randomly samples a proportion 'p' of rows of a dataframe
    '''
    size = df.shape[0]
    np.random.seed(seed)
    return df.ix[np.random.randint(0, size, int(size*p)), :]

def stats_df(df):
    '''
    Gives stats about the dataframe
    '''
    print("Nb lines in the train : ",len(df["from"]))
    print("Nb of unique nodes : ",len(df["from"].unique()))
    print("The document that cites the most, cites : ",df.groupby(["from"]).sum()["y"].max()," document(s).")
    print("The document with no citation : ",sum(df.groupby(["from"]).sum()["y"] == 0),"\n")
    print("The most cited document, is cited : ",df.groupby(["to"]).sum()["y"].max()," times.")
    print("Nb of documents never cited  : ",sum(df.groupby(["to"]).sum()["y"] == 0),"\n")
    print("There are NaN to handle for authors and journalName :")