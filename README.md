# Movie-genre-detector

## Abstract
Deep Learning shallow network for genre detection via tfidf NLP techinque.

## Dataset
ML Detector based on [MovieLens dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv)

## Requirements
- python == 3.8.5
- tensorflow==2.2.0 for Model build and training
- scikit-learn==0.24.2 for advanced data preprocessing techinques for NLP (e.g. tfidf)
- pandas==1.2.4 for data ingestion and management /n
For the other requirements check out requirements.txt

## Installation
For Docker install check out [official documentation](https://docs.docker.com/get-docker/)
$ git clone https://github.com/andrea-t94/Movie-genre-detector.git
$ docker build -t ml-model -f Dockerfile .

 
