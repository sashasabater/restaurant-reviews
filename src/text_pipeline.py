import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



class TextPrep:
    
    def __init__(self, df):
        #initialize with dataframe
        self.df = df
        self.documents = []
        #creating a documents attribute
        self.feature_names = []
        #creating a feature names attribute(bag of words)
    
    def vectorize(self,column, method='tfidf'):
        self.target = column
        #takes in name of column to be cleaned, and method to vectorize
        lem = WordNetLemmatizer()
        #lemmatizing as opposed to stemming 
        s_words = stopwords.words('english')
        texts = []
        for doc in self.df[column].values:
            cleaned = [lem.lemmatize(word).lower() for word in doc.split(' ') \
                       if word not in s_words and word.isalpha() == True]
            #making sure that every word only has letters
            texts.append(' '.join(cleaned))
        self.documents = texts 
  
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            vectorized = vectorizer.fit_transform(self.documents).toarray()
            self.feature_names = vectorizer.get_feature_names()
            return vectorized
            
        elif method == 'count':
            vectorizer = CountVectorizer(stop_words='english', max_features=5000)
            vectorized = vectorizer.fit_transform(self.documents).toarray()
            self.feature_names = vectorizer.get_feature_names()
            return vectorized
        elif method == 'token':
            puncs = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
            vectorizer = Tokenizer(num_words=50000, filters =puncs, lower=True)
            vectorizer.fit_on_texts(self.df[self.target].values)
            self.feature_names = vectorizer.word_index
            vectorized = vectorizer.texts_to_sequences(self.df[self.target].values)
            vectorized = pad_sequences(vectorized, maxlen=250)
            return vectorized
            #returns final vectorized feature matrix for modeling 
    
    