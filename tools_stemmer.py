import nltk

nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
#stemmer = nltk.stem.PorterStemmer()


def remove_stopwords_and_stem(words):
    words = [token for token in words if (len(token) > 2 and (token not in stpwds))]
    return [token for token in words] #stemmer.stem(token)

