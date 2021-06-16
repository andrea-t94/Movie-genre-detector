#!/usr/bin/python3
# inference.py

import argparse
import warnings
warnings.filterwarnings("ignore")

from genre_detector import BaselineModel
from text_processor import TextProcessor

def infer():
    #parse input data
    parser = argparse.ArgumentParser(description='Movie genre classification')
    parser.add_argument('--title', metavar='title', required=True, type=str, help='title of a movie')
    parser.add_argument('--description', metavar='description', required=True, type=str,
                        help='descriprion of the movie')
    args = parser.parse_args()
    title = args.title
    description = args.description
    print(title)

    #model selection
    try:
        model = BaselineModel(name = "genre-detector")
        model.load_best_model()
        print("Model successfully loaded, with a tested accuracy of {}".format(model.best_acc))
    except:
        raise FileNotFoundError("No model found, please run the train.py before!")

    #text processor load
    try:
        processor = TextProcessor()
        processor.load_processor()
        print("Processor successfully loaded")
    except:
        raise FileNotFoundError("No encoder found, something went wrong with train.py")


    #input preprocessing
    text = [title + " " + description] #processor tfiff need iterable
    tfidf_text = processor.tfidf_process(text)

    #inference
    prediction = model.class_prediction(tfidf_text)
    class_name = processor.decode_label(prediction) #expected array of shape (1,) since input of one record
    output = {"title": title,
              "description": description,
              "genre": class_name.item()}
    print(output)

if __name__ == '__main__':
    infer()

