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

similarity = scipy.sparse.load_npz('Data/similarity.npz')

print(similarity[1,:])
print("Hi")
print(similarity[2,:])
print(similarity[3,:])

print(similarity[2,84])
