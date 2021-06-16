#!/usr/bin/python3
# test.py

import os

#test text processor correctly saved
try:
    assert os.path.exists("artifacts")
except:
    raise NotADirectoryError("TextProcessor storage directory not found, something went wrong with data processing")

#test model correctly saved
try:
    assert os.path.exists("models")
except:
    raise NotADirectoryError("Model storage directory not found, something went wrong with training")
