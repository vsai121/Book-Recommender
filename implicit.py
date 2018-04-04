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

sparseMatrix = scipy.sparse.load_npz('Data/simBaseline.npz')

similarity = sparseMatrix.todense()

print(np.count_nonzero(similarity==1))
