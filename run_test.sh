#!/bin/bash

python3 test.py

echo Text processor and Model correctly built

python3 inference.py --title "Othello" --description \
      "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic."

echo test passed!