import pandas as pd
import numpy as np
from ast import literal_eval
from pandas.io.json import json_normalize

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


#raw data ingestion
col_list = ['genres','original_title','overview']
dataset_path = '/Users/andreatamburri/Desktop/MovieLensDataset/movies_metadata.csv'
df_movielens_raw = pd.read_csv(dataset_path, usecols=col_list)

#data preprocessing
df_movielens_prep = df_movielens_raw.copy()
df_movielens_prep['genre'] = df_movielens_prep.apply(lambda row: extract_first_json(row['genres']), axis=1)
df_movielens_prep.dropna(inplace=True)
df_movielens_prep[['genre_id','genre_name']] = json_normalize(df_movielens_prep['genre'])

#data labeling
#df_movielens_raw.genres = df_movielens_raw.genres.apply(literal_eval)

#TODO
##tfidf preprocessing
##stopwords and data cleansing
##save 10% for testing
##traintest split kfold
##score definition
##score logs in wandb
##model saving
##inference part with test
##automated testing
##production ready
##dockerization with datased already saved
##readme
##code cleanup

print(df_movielens_prep.info())
print(df_movielens_prep.head(10))