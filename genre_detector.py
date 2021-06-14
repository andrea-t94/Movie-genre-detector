import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import accuracy_score,precision_score, recall_score
from keras.models import load_model
import warnings

#TODO
#set LOCAL_DIR as env var in Docker
from configs import LOCAL_DIR

class baselineModel(object):
    def __init__(self, name, uuid = None):
        self.name = name
        self.uuid = str(uuid)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        if uuid:
            os.makedirs("models/" + self.name + "/" + self.uuid)

    def build(self, input_shape, output_shape, config):
        #define the parameters
        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape
        #build the model
        input_layer = Input(shape=self.input_shape)
        x = Dense(self.config["dense"], activation='relu')(input_layer)
        x = Dropout(self.config["dropout"])(x)
        out = Dense(self.output_shape, activation='softmax')(x)
        self.model = Model(name='Genre-detector', inputs=input_layer, outputs=out)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"]
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=['accuracy']
        )
        self.model.summary()

    def train(self, x_train, y_train, validation):
        self.model.fit(
            x_train,
            y_train,
            validation_data=validation,
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            callbacks=[
                self.log_callback,
                tf.keras.callbacks.ModelCheckpoint(
                    filepath= LOCAL_DIR + "models/" + self.name + "/" + self.uuid + "/{val_accuracy:.2f}.h5",
                    monitor='val_loss', save_best_only=True, verbose=1)
            ]
        )

    def test(self, x_test, y_test, return_score=True):
        # predict crisp classes for test set
        yhat_classes = np.argmax(self.model.predict(x_test, verbose=0), axis=1)
        Y_test = np.argmax(y_test, axis=-1)
        acc = round(accuracy_score(Y_test, yhat_classes), 4) * 100
        print('Accuracy: %f' % acc)
        precision = round(precision_score(Y_test, yhat_classes, average='weighted'), 4) * 100
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = round(recall_score(Y_test, yhat_classes, average='weighted'), 4) * 100
        print('Recall: %f' % recall)
        score = {
            "Accuracy %": acc,
            "Precision %": precision,
            "Recall %": recall,
        }
        if return_score:
            return score
        else:
            print("Scores on test data: " + score)

    def load_best_model(self):
        #directory in which store models with unique uuid
        #LOCAL_DIR to be discarded
        models_dir =  LOCAL_DIR + "models/" + self.name + "/"
        self.best_acc = 0
        self.best_model_path = None
        for dirs in os.listdir(models_dir):
            tmp_models_subdir = models_dir + dirs + "/"
            print(tmp_models_subdir)
            try:
                for model_file in os.listdir(tmp_models_subdir):
                    model_acc = float(os.path.splitext(model_file)[0])
                    if model_acc > self.best_acc:
                        self.best_acc = model_acc
                        self.best_model_path = tmp_models_subdir + model_file
                        print(self.best_model_path)
            except:
                warnings.warn("Problems with directory {}".format(tmp_models_subdir))
        #TODO
        # test if model exists
        if self.best_model_path:
            self.model = load_model(self.best_model_path)
            print("Model loaded with accuracy: {}".format(self.best_acc))
        else:
            raise FileNotFoundError("No trained model found in {}".format(models_dir))
