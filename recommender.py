import pandas as pd
import numpy as np

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import KNNBasic
from surprise.accuracy import accuracy.mae

shape =  (12001 , 10001)

br_cols = ['book_id' , 'user_id' , 'rating']
bookRatings = pd.read_csv('Data/ratings.csv' , sep=',' , names = br_cols , encoding='latin-1' , low_memory=False , skiprows=[0])

print(bookRatings.head())

#print(ratings.head())

b_cols = ['book_id' , 'books_count' , 'isbn' , 'isbn13' , 'authors' , 'original_publication_year' , 'original_title' , 'average_rating' , 'ratings_count' , 'image_url']
books = pd.read_csv('Data/books.csv' , sep=',' , usecols=b_cols , encoding='latin-1' , low_memory = False)

print(books.head())

tr_cols = ['user_id' , 'book_id']
to_read = pd.read_csv('Data/to_read.csv' , sep=',' , usecols=tr_cols , encoding='latin-1' , low_memory = False)
#print(books.head())

print(bookRatings.shape)
print(books.shape)
print(to_read.shape)


bookRatings = bookRatings.drop_duplicates(['user_id' , 'book_id'] , 'first')
bookRatings.groupby('user_id').filter(lambda x: len(x) >= 4)

print(bookRatings.shape)

bookRatings = bookRatings.sample(20000)
#print(bookRatings.iloc[6])
print(bookRatings.shape)


reader = Reader(rating_scale=(1, 5) )
data = Dataset.load_from_df(bookRatings , reader)

trainingSet = data.build_full_trainset()
print(trainingSet)


sim_options = {

'name': 'cosine',
'user_based': False
}


knn = KNNBasic(sim_options=sim_options)
knn.fit(trainingSet)

testSet = trainingSet.build_anti_testset()
predictions = knn.test(testSet)

print(accuracy.fcp(predictions))
