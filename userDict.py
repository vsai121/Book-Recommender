
import pandas as pd
import numpy as np

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import KNNBasic , KNNWithMeans , KNNWithZScore , KNNBaseline
from surprise import accuracy
from surprise.model_selection import train_test_split

import scipy.sparse

userDict={}
br_cols = ['book_id' , 'user_id' , 'rating']
bookRatings = pd.read_csv('Data/ratings.csv' , sep=',' , names = br_cols , encoding='latin-1' , low_memory=False , skiprows=[0])
bookRatings = bookRatings[['user_id' , 'book_id' , 'rating']]

print(bookRatings.head())

for row in bookRatings.iterrows():

    book_id = row[1][1]
    user_id = row[1][0]
    rating = row[1][2]

    l = [book_id,rating]

    if userDict.has_key(user_id):
        l1 = userDict.get(user_id)
        l1.append(l)
        userDict[user_id]=l1

    else:
        userDict[user_id] = [l]


print(userDict[1])
np.save('Data/userDict.npy' , userDict)
