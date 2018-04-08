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

print("Finally working I guess pearson")

sparseMatrix = scipy.sparse.load_npz('Data/sim.npz')
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

List=[]
oldID = -1
rowlist=[]
cnt = 0

print("Saving dataframe bro pls")

for row in to_read.iterrows():

    dict1 = dict()
    user_id = row[1][0]
    book_id = row[1][1]

    calcRating=0
    denum=0
    average_rating=0
    denum2=0
    dist = 0

    if user_id != oldID:
        #print(user_id)
        List = userDict[user_id]
        oldID = user_id
        #print(List)

    for item in List:
        compBookID = item[0]
        rating = item[1]
        #print(user_id , book_id , compBookID)

        if book_id<10000 and compBookID <10000:
            sim = similarity[book_id , compBookID]


            average_rating = books.loc[compBookID]['average_rating']
            dist += (average_rating - rating)**2
            denum2+=1

            if sim>0.5:
                calcRating+= sim*rating
                denum+=sim

    if denum!=0:
        cnt+=1
        print(cnt)
        calcRating/=denum
        #print(calcRating)
        if calcRating!=0:
            dict1.update({'user_id':user_id , 'book_id':book_id , 'rating':randint(3,5)})
            #print(dict1)
            rowlist.append(dict1)
            #print(rowlist)

    else:
        if book_id <10000:
            dist/=denum2
            if dist<=0.5:
                avg = books.loc[book_ID]['average_rating']
                dict1.update({'user_id':user_id , 'book_id':book_id , 'rating':avg})



#print(rowlist)
print("Saving now")
df = pd.DataFrame(rowlist)
bookRatings = bookRatings.append(df , ignore_index=True)

print(bookRatings.shape)

bookRatings.to_csv('Data/implicitRatingsPearson.csv' , sep=',')
