import re
import unicodedata
import nltk

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

from configs import LOCAL_DIR

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, text):
        stop_words = nltk.corpus.stopwords.words('english')
        newStopWords = ['--', 'le','u']  # It's common in the dataset, not informative
        stop_words.extend(newStopWords)
        text = (unicodedata.normalize('NFKD', text)
                .encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
        text = re.sub(r'[^\w\s]', ' ', text)  # cleanup strange characters
        stems = [self.wnl.lemmatize(t) for t in word_tokenize(text) if not t in stop_words]
        return stems


class TextProcessor:
    ''' Preprocess text using TfidfVectorizer and LabelEncoder '''

    def __init__(self,
                 analyzer: Optional[str] = 'word',
                 stop_words: Optional[str] = 'english',
                 strip_accents: Optional[str] = 'unicode',
                 lowercase: Optional[bool] = True,
                 lemmatizer: Optional[bool] = True
                 ):
        self.analyzer = analyzer
        self.stop_words = stop_words
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.lemmatizer = lemmatizer

    def tfidf_process(self, input):
        if hasattr(self, 'tfidf'):
            output = self.tfidf.fit_transform(input).toarray()
        else:
            #build from scratch
            if self.lemmatizer == True:
                self.tokenizer = LemmaTokenizer()
            else:
                self.tokenizer = None
            self.tfidf = TfidfVectorizer(analyzer=self.analyzer,
                                    stop_words=self.stop_words,
                                    strip_accents=self.strip_accents,
                                    lowercase=self.lowercase,
                                    tokenizer=self.tokenizer)
            output = self.tfidf.fit_transform(input).toarray()
        return output

    def encode_label(self, input):
        if hasattr(self, 'encoder'):
            output = self.encoder.transform(input)
        else:
            #build from scratch
            self.encoder = LabelEncoder()
            output = self.encoder.fit_transform(input)
        return output

    def decode_label(self, input):
        if hasattr(self, 'encoder'):
            output = self.encoder.inverse_transform(input)
        else:
            raise ValueError("No loaded encoder found, please encode labels before decoding")
        return output

    def save_processor(self, filepath: Optional[str] = LOCAL_DIR + '/artifacts/'):
        with open(filepath + 'tfidf_vectorizer.pkl', 'wb') as fw:
            dump(self.tfidf.vocabulary_, fw)
        with open(filepath + 'label_encoder.pkl', 'wb') as fw:
            dump(self.encoder, fw)

    def load_processor(self, filepath: Optional[str] = LOCAL_DIR + '/artifacts/'):
        self.encoder = load(filepath + "label_encoder.pkl")
        tfidf_dictionary = load(filepath + "tfidf_vectorizer.pkl")
        if self.lemmatizer == True:
            self.tokenizer = LemmaTokenizer()
        else:
            self.tokenizer = None
        self.tfidf = TfidfVectorizer(analyzer=self.analyzer,
                                     stop_words=self.stop_words,
                                     strip_accents=self.strip_accents,
                                     lowercase=self.lowercase,
                                     tokenizer=self.tokenizer,
                                     vocabulary=tfidf_dictionary)
        return self.tfidf, self.encoder





