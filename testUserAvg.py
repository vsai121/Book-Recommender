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

from random import randint

print("Should work now")

sparseMatrix = scipy.sparse.load_npz('Data/simCosine.npz')
similarity = sparseMatrix.todense()

userDict = np.load('Data/userDict.npy').item()

b_cols = ['book_id' , 'books_count' , 'isbn' , 'isbn13' , 'authors' , 'original_publication_year' , 'original_title' , 'average_rating' , 'ratings_count' , 'image_url']
books = pd.read_csv('Data/books.csv' , sep=',' , usecols=b_cols , encoding='latin-1' , low_memory = False)


tr_cols = ['user_id' , 'book_id']
to_read = pd.read_csv('Data/to_read.csv' , sep=',' , usecols=tr_cols , encoding='latin-1' , low_memory = False)

br_cols = ['book_id' , 'user_id' , 'rating']

bookRatings = pd.read_csv('Data/ratings.csv' , sep=',' , names = br_cols , encoding='latin-1' , low_memory=False , skiprows=[0])
bookRatings = bookRatings[['user_id' , 'book_id' , 'rating']]

bookRatings = bookRatings.drop_duplicates(['user_id' , 'book_id'] , 'first')
bookRatings.groupby('user_id').filter(lambda x: len(x) >= 4)


print(to_read.head())
print(bookRatings.shape)


oldID = -1
rowlist=[]

print("Saving dataframe bro pls")

for row in to_read.iterrows():

    dict1 = dict()
    user_id = row[1][0]
    book_id = row[1][1]
    chk=0

    if user_id!=oldID:
        oldID = user_id
        avgRat = 0.0
        totalRat=0
        cnt=0
        List = userDict[user_id]
        chk=1


    if book_id <10000:

        if chk==1:
            for book in List:
                totalRat+=book[1]
                cnt+=1

            avgRat = totalRat/cnt


        calcRating = avgRat
        print(calcRating)
        dict1.update({'user_id':user_id , 'book_id':book_id , 'rating':calcRating})
        #print(dict1)
        rowlist.append(dict1)
        #print(rowlist

#print(rowlist)
print("Saving now")
df = pd.DataFrame(rowlist)
bookRatings = bookRatings.append(df , ignore_index=True)

print(bookRatings.shape)

bookRatings.to_csv('Data/implicitRatingsCosine.csv' , sep=',')
