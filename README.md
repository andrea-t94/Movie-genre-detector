# Movie-genre-detector

## Abstract
Deep Learning shallow network for genre detection via tfidf NLP techinque.

## Dataset
ML Detector based on [MovieLens dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv)

## Requirements
- python == 3.8.5
- tensorflow==2.2.0 for Model build and training
- scikit-learn==0.24.2 for advanced data preprocessing techinques for NLP (e.g. tfidf)
- pandas==1.2.4 for data ingestion and management\
For the other requirements check out requirements.txt

## Project file structure

```
/MovieLensDataset
    /.csv
/Dockerfile
/train.py
/inference.py
/test.py
/models #in which stored model trained
/artifacts #in which stored text processor trained
/...
```

## Installation
For Docker install check out [official documentation](https://docs.docker.com/get-docker/)
- clone repository ```
$ git clone https://github.com/andrea-t94/Movie-genre-detector.git ```
- build docker image ```
$ docker build -t ml-model -f Dockerfile . ```\
\
Docker build will:
- install all the dependencies
-  run a training for deploying the first model (NB. the train will deal with the first model and processor deployment)
-  test if the build was correctly deployed\
#### NB. The build could fail due to OOM, for testing purpose it's advisable to decrease the max_features parameters of the TextProcessor instance in the training)

## How to run
Once built the docker image:
- ```$ docker run ml-model python3 train.py ``` will handle the training and store the new text processor in artifacts directory and the model in model directory
- ``` $ docker run ml-model python3 inference.py --title "title" --description "description" ``` will return the genre of the movie
- ``` $ docker run ml-model bash ./run_test.sh ``` will run the automated test (in which there is an example of inference)



 
