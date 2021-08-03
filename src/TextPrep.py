import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



class TextPrep:

	def __init__(self, df):
		#initializing with a dataframe
		self.df = df 
	
	def clean(self,column):
		#the clean method takes in the name of the column to be cleaned
		lem = WordNetLemmatizer()
		stopwords = stopwords.words('english')
		texts = []
		for doc in self.df[column].values:
			cleaned = [lem.lemmatize(word).lower() for word in doc.split(' ') \
				if word not in stopwords and word.isalpha()==True]
		texts.append(" ".join(cleaned))
		self.documents = texts 
		return self.documents
	def vectorize(self, method='tfidf'):
		#vectorize according to method, either tfidf or count 
		if method == 'tfidf':
			vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
			#returns feature matrix X read for modeling 
			self.X = vectorizer.fit_transform(self.documents).toarray()
			self.feature_names = vectorizer.get_feature_names()
		elif method == 'count':
			vectorizer = CountVectorizer(stop_words='english', max_features=5000)
			self.X = vetorizer.fit_transofmr(self.documents).toarray()
			self.feature_names = vectorizer.get_feature_names()
		return self.X 


