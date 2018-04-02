
import numpy as np
from six import iteritems

def cosine(n_x, yr, min_support):
    """Compute the cosine similarity between all pairs of users (or items).
    """

    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)


    i = 0
    bid=[]
    ratings=[]

    for y , y_ratings in iteritems(yr):
        bid = (y_ratings.index).tolist()
        ratings = (y_ratings.values).tolist()

        for xi ,ri in zip(bid , ratings):
            print("Entered")
            print(xi)
            print(ri)
            for xj, rj in zip(bid ,  ratings):
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_support:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return sim
