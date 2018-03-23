import pandas as pd
import numpy as np


shape =  (12001 , 10001)

br_cols = ['book_id' , 'user_id' , 'rating']
bookRatings = pd.read_csv('Data/ratings.csv' , sep=',' , names = br_cols , encoding='latin-1' , low_memory=False)

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

#print(bookRatings.iloc[6])
print(bookRatings.shape)

ratings = np.zeros(shape)
print("First ratings shape")
print(ratings.shape)
#print(bookRatings)
i = 0
for row in bookRatings.itertuples():
    if i==0:
        i=1
        continue
    else:
        if(int(row[2])<=12000):
            ratings[int(row[2]) , int(row[1])] = int(row[3])

#print("Minimum is " + str(min))
print(np.count_nonzero(ratings[6,:]))
#df = pd.DataFrame(ratings)
#df.to_csv('Data/ratingsTable.csv' , sep=',')

#print(ratings)

"""
toRead = np.zeros((100000 , shape[1]))

i = 0

rows = toRead.sample(100000)

for row in rows.itertuples():
    toRead[int(row[1]) , int(row[2])] = int(1)

df = pd.DataFrame(toRead)
df.to_csv('goodbooks-10k/toRead.csv')
"""

print("Opened ratings")
