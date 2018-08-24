import pandas as pd


def load_r_data(filename):
    d2 = pd.read_csv(filename, sep=";")
    d2.drop('Unnamed: 0', axis=1, inplace=True)
    return d2
