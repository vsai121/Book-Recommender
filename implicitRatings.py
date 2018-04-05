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

print("Yussss")

br_cols = ['book_id' , 'rating' , 'user_id']
bookRatings = pd.read_csv('Data/implicitRatingsCosine.csv' , sep=',' , names = br_cols , encoding='latin-1' , low_memory=False , skiprows=[0])
bookRatings = bookRatings[['user_id' , 'book_id' , 'rating']]

reader = Reader(rating_scale=(1, 5) )
data = Dataset.load_from_df(bookRatings , reader)

trainingSet, testSet = train_test_split(data, test_size=.15)

sim_options = {
'name': 'cosine',
'user_based': False,
'min_support': 3

}

knn = KNNBasic(k = 100 , min_k = 3 ,sim_options=sim_options)

knn.fit(trainingSet)

predictions = knn.test(testSet)
print(accuracy.rmse(predictions))
