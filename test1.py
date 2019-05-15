import pandas as pd

train = pd.read_csv('mnist_train.csv')
train = train.values
print train