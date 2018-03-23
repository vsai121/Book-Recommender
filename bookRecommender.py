import pandas as pd
import numpy as np


shape =  (12001 , 10001)

from loaddata import *

print("Opened ratings")
#print(ratings)


def train_test_split(ratings):
    print("Rating shape")
    print(ratings.shape)
    test = np.zeros(shape)
    train = ratings.copy()
    i=1
    for user in xrange(1,shape[0]):
        if(np.count_nonzero(ratings[user,:])>=3):
            print(str(i))
            i+=1
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=1,
                                        replace=True)
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]

    return train , test

def fast_similarity(ratings, kind='user', epsilon=1e-9):
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    print(norms.shape)
    return (sim /(norms.dot(norms.T)))

def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

print("Here first")
train , test = train_test_split(ratings)
user_similarity = fast_similarity(train, kind='user')
print(user_similarity)

item_similarity = fast_similarity(train , kind='item')
print(item_similarity.shape)

user_prediction = predict_fast_simple(train, user_similarity, kind='user')
item_prediction = predict_fast_simple(train , item_similarity, kind='item')

print 'User-based CF MSE: ' + str(get_mse(user_prediction, test))
print 'Item-based CF MSE: ' + str(get_mse(item_prediction, test))
