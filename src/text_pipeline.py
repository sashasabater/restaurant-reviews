import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TextPrep:
    
    def __init__(self, df):
        #initialize with dataframe
        self.df = df
    
    def vectorize(self,column, method='tfidf'):
        #takes in name of column to be cleaned, and method to vectorize
        lem = WordNetLemmatizer()
        #lemmatizing as opposed to stemming 
        s_words = stopwords.words('english')
        texts = []
        for doc in self.df[column].values:
            cleaned = [lem.lemmatize(word).lower() for word in doc.split(' ') \
                       if word not in s_words and word.isalpha() == True]
            #making sure that every word onl
            texts.append(' '.join(cleaned))
        self.documents = texts 
  
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            vectorized = vectorizer.fit_transform(self.documents).toarray()
            return vectorized
            #creating a feature names attribute(bag of words)
        elif method == 'count':
            vectorizer = CountVectorizer(stop_words='english', max_features=5000)
            vectorized = vectorizer.fit_transform(self.documents).toarray()
            return vectorized
            #returns final vectorized feature matrix for modeling 
    
    