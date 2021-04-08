import numpy as np
import pandas as pd


def get_rating_matrix(filename, dtype=np.float32):
    data = pd.read_csv(filename)
    return data.groupby(['source','target'])['rating'].first().unstack().fillna(0)

def get_frequent_matrix(filename,dtype=np.float32) :
    data = pd.read_csv(filename)
    data['rating']=1
    return data.groupby(['source','target'])['rating'].sum().unstack().fillna(0)
