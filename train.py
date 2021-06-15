#!/usr/bin/python3
# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import uuid

from genre_detector import BaselineModel
from text_processor import TextProcessor
from configs import model_config, movielens_col_list
from helpers import extract_first_json

def train():
    #raw data ingestion
    dataset_path = 'MovieLensDataset/movies_metadata.csv'
    df_movielens_raw = pd.read_csv(dataset_path, usecols=movielens_col_list)
    print(df_movielens_raw.head())

    #raw data preprocessing
    print("data preprocessing")
    df_movielens_prep = df_movielens_raw.copy()
    df_movielens_prep['genre'] = df_movielens_prep.apply(lambda row: extract_first_json(row['genres']), axis=1)
    df_movielens_prep.dropna(inplace=True)
    df_movielens_prep[['genre_id','genre_name']] = pd.json_normalize(df_movielens_prep['genre'])
    df_movielens_prep['text'] = df_movielens_prep.apply(
        lambda row: row['original_title'] + ' ' + row['overview'], axis=1) #considering overview and title as unique text
    df_movielens_final = df_movielens_prep[['text', 'genre_id', 'genre_name']].copy()
    df_movielens_final.dropna(inplace=True)

    X_text = df_movielens_final['text']
    y = df_movielens_final['genre_name']
    #nlp preprocessing and data labeling
    processor = TextProcessor()
    X_tfidf = processor.tfidf_process(X_text)
    y_encoded = processor.encode_label(y)
    num_class = len(pd.unique(df_movielens_final['genre_name']))
    y_categorical = to_categorical(y_encoded, num_classes=num_class)

    #save processor for further use in inference
    processor.save_processor()

    #training testing dataset prep
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y_categorical, test_size=0.2, random_state=0)

    #model creation
    model = BaselineModel(name = "genre-detector", uuid = uuid.uuid1())
    input_shape = (X_tfidf.shape[1])
    model.build(input_shape=input_shape, output_shape = num_class, config= model_config)
    model.train(X_train, y_train, (X_test, y_test))
    score = model.test(X_test, y_test)
    print(score)

if __name__ == '__main__':
    train()


