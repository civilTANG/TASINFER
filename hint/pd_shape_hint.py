import pandas as pd
path = 'D:\dataset\digit-recognizer\\train.csv'

def read_csv(path):
    shape = pd.read_csv(path).shape
    demin = len(shape)
    return [demin, shape, []]


con = read_csv(path)

print(con)

