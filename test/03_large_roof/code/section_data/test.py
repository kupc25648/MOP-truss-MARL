import os, sys
import csv
import numpy as np
import pandas as pd

def read_column(path):

    #df = pd.read_csv(path)
    #print(df)
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    data = np.array(data)
    data = data.astype(int)

    return data

read_data = read_column(path)

