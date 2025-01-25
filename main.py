
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from numpy.ma.extras import average
from sklearn.linear_model import LinearRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
arr=np.array([[1,2,3]])
arr2=np.array([[4,5,6]])

similarity=cosine_similarity(arr,arr2)

set_a=[0,1,1,1,0]
set_b=[1,1,0,1,1]
similarityJ=jaccard_score(set_a,set_b,average='binary')
print("Jaccard Similarity:",similarityJ)
print("Cosine similarity:",similarity[0][0])

X=np.array([[500],[1000],[1500],[2000],[2500]])
y=np.array([150000, 300000, 450000, 600000, 750000])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
