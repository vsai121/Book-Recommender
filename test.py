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

print("Cosine pls work")

sparseMatrix = scipy.sparse.load_npz('Data/simCosine.npz')
similarity = sparseMatrix.todense()

userDict = np.load('Data/userDict.npy').item()

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

            average_rating+=rating
            denum2+=1

            if sim>0.5:
                calcRating+= sim*rating
                denum+=1

    if denum!=0:
        cnt+=1
        print(cnt)
        calcRating/=denum
        #print(calcRating)
        if calcRating!=0:
            dict1.update({'user_id':user_id , 'book_id':book_id , 'rating':calcRating})
            #print(dict1)
            rowlist.append(dict1)
            #print(rowlist)

    else:
        if denum2!=0:
            average_rating/=denum2
            dict1.update({'user_id':user_id , 'book_id':book_id , 'rating':average_rating})
            rowlist.append(dict1)

#print(rowlist)
print("Saving now")
df = pd.DataFrame(rowlist)
bookRatings = bookRatings.append(df , ignore_index=True)

print(bookRatings.shape)

bookRatings.to_csv('Data/implicitRatingsCosine.csv' , sep=',')
