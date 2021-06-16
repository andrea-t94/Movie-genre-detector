FROM python:3.8.5

#install requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet

#copy dataset
COPY MovieLensDataset/movies_metadata.csv ./MovieLensDataset/movies_metadata.csv

#copy scripts
COPY train.py ./train.py
COPY test.py ./test.py
COPY inference.py ./inference.py
COPY configs.py ./configs.py
COPY genre_detector.py ./genre_detector.py
COPY helpers.py ./helpers.py
COPY text_processor.py ./text_processor.py

#train and deploy first model
RUN mkdir -p artifacts/
RUN python3 train.py

#automatic tests
COPY run_test.sh ./run_test.sh
RUN bash ./run_test.sh
CMD ./run_test.sh
