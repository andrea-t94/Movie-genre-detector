import numpy as np
from ast import literal_eval
import nltk
import unicodedata
import re
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

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

def extract_first_json(col: str):
    ''' The function takes as input a string of multiple JSON files
    and return the first JSON object'''
    #check if empty
    if col == '[]':
        return np.nan
    else:
        col_list = literal_eval(col)
        first_json = col_list[0]
        return first_json

def basic_clean(text):
    ''' basic nlp cleanup'''
    #TODO
    #to download once
    #nltk.download('stopwords')
    #nltk.download('wordnet')
    nltk.download('punkt')
    if len(text) == 0:
        return ''
    wnl = nltk.stem.WordNetLemmatizer()
    stop_words = nltk.corpus.stopwords.words('english')
    newStopWords = ['--'] #It's common in the dataset, not informative
    stop_words.extend(newStopWords)
    text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    text = re.sub(r'[^\w\s]', ' ', text) #cleanup strange characters
    words = text.split()
    words = [wnl.lemmatize(word) for word in words if not word in stop_words]
    return words