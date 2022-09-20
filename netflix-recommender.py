import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("netflixData.csv")

# drop rows containing null values 
data = data.dropna()
print(data.head()) #checking the first 5 rows of data
print(data.isnull().sum()) #checking for null values in each column 