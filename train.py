import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pandas.core.algorithms import mode

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]


def convert_to_int(word):
    word_dict = {'one':1 ,'two':2 ,'three':3 ,'four':4 ,'five':5 ,'six':6 ,'seven':7 ,'eight':8 ,
                'nine':9 ,'ten':10 ,'eleven':11 ,'twelve':12 ,'thirteen':13 , 'zero':0 , 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Spliting Training and Test Set 
#Since we have a very small Dataset, we will train our model with all available data

from sklearn.linear_model import LinearRegression
r = LinearRegression()

#Fitting model with training data
r.fit(X, y)

#Saving model to disk
pickle.dump(r, open('model.pkl', 'wb'))

#Loading model to compare results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2, 9, 6]]))
