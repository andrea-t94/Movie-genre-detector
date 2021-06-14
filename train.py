import pandas as pd
import numpy as np
from ast import literal_eval
from pandas import json_normalize
import nltk
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import uuid

from genre_detector import baselineModel
from configs import model_config

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
    if len(text) == 0:
        return ''
    wnl = nltk.stem.WordNetLemmatizer()
    stop_words = nltk.corpus.stopwords.words('english')
    #stop_words.remove('non')
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


#raw data ingestion
col_list = ['genres','original_title','overview']
dataset_path = '/Users/andreatamburri/Desktop/MovieLensDataset/movies_metadata.csv'
df_movielens_raw = pd.read_csv(dataset_path, usecols=col_list)

#data preprocessing
df_movielens_prep = df_movielens_raw.copy()
df_movielens_prep['genre'] = df_movielens_prep.apply(lambda row: extract_first_json(row['genres']), axis=1)
df_movielens_prep.dropna(inplace=True)
df_movielens_prep[['genre_id','genre_name']] = json_normalize(df_movielens_prep['genre'])

print('text cleaning and preprocessing')
df_movielens_prep['text'] = df_movielens_prep.apply(lambda row: row['original_title'] + ' ' + row['overview'], axis=1)
df_movielens_prep['clean_body'] = df_movielens_prep['text'].apply(basic_clean)

df_movielens_final = df_movielens_prep[['clean_body', 'genre_id', 'genre_name']]
df_movielens_final.dropna(inplace=True)
df_movielens_final.to_csv(
        r'/Users/andreatamburri/Desktop/MovieLensDataset/test_dataset.csv')

X_model = df_movielens_final['clean_body'].apply(lambda x: ' '.join(x))
y_model = df_movielens_final['genre_id'].apply(lambda x: int(x))
# convert integers to dummy variables (i.e. one hot encoded)
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_model)
encoded_Y = encoder.transform(y_model)
dummy_y = np_utils.to_categorical(encoded_Y, num_classes=23)
print('tfidf preprocessing')
tfidf = TfidfVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None)
tf_idf_X = tfidf.fit_transform(X_model).toarray()
df_tfidfvect = pd.DataFrame(data = tf_idf_X)
input_shape = (tf_idf_X.shape[1],)
print(tf_idf_X.shape)

#training testing dataset prep
print("train test split")
X_train, X_test, y_train, y_test = train_test_split(
    df_tfidfvect, dummy_y, test_size=0.2, random_state=0)

#data labeling
num_class = len(pd.unique(df_movielens_final['genre_name']))

#model
model = baselineModel(name = "genre-detector", uuid = uuid.uuid1())
model.build(input_shape=input_shape, output_shape = num_class, config= model_config)
model.train(X_train, y_train, (X_test, y_test))
score = model.test(X_test, y_test)
print(score)

#TODO
##mapper genre_name genre_label_id
##dependencies management, e.g. nltk import and download stopwords
##save 10% for testing
##set LOCAL_DIR as env var in Docker
##if genre not found return message not found genre
##inference part with test
##automated testing
##dockerization with datased already saved
##readme
##code cleanup


