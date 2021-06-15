import numpy as np
from ast import literal_eval

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
