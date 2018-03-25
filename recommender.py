import pandas as pd
import numpy as np

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import KNNBasic
from surprise import accuracy
from surprise.model_selection import train_test_split


from collections import defaultdict

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

bookRatings = bookRatings[bookRatings['user_id']<=15000]
print(bookRatings.shape)


reader = Reader(rating_scale=(1, 5) )
data = Dataset.load_from_df(bookRatings , reader)

trainingSet, testSet = train_test_split(data, test_size=.15)

sim_options = {
'name': 'pearson_baseline',
'user_based': False
}


knn = KNNBasic(k = 50 , min_k = 1 ,sim_options=sim_options)
knn.fit(trainingSet)

predictions = knn.test(testSet)

#print(predictions)

#Prediction(uid=6727, iid=9476, r_ui=3.0, est=3.0, details={u'actual_k': 1, u'was_impossible': False}),

print("Running the loop now \n")
for bid , uid , rui , est , details in predictions:

    if(details['was_impossible']==True):
        continue

    else:
        if details['actual_k'] >=5 :

            print(bid , uid , rui , est)


def get_topN_recommendations(predictions , topN=5):

    top_recs = defaultdict(list)

    for uid, iid, true_r, est, details in predictions:


         if(details['was_impossible']==False):
             continue

         top_recs[iid].append((uid, est))

    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[iid] = user_ratings[:topN]


    return top_recs


top3_recommendations = get_topN_recommendations(predictions , topN=3)
for uid, user_ratings in top3_recommendations.items():
    for iid , _ in user_ratings:

        print(iid , books.loc[uid]['original_title'])





print(accuracy.rmse(predictions))
