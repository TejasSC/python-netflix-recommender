import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

import nltk
import re
#nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

data = pd.read_csv("netflixData.csv")
#print(data.head()) #checking the first 5 rows of data
# drop rows containing null values 
data = data.dropna()

# clean function to remove stopwords, unnecessary punctuation, non-alphanumeric characters etc. 
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

data["cleanTitle"] = data["Title"].apply(clean) #use clean function on Title column of movies 

#print(data.head()) 
#use genres as feature to recommend similar content to user
feature = data["Genres"].tolist()
tfidf = text.TfidfVectorizer(
    input=feature,
    stop_words="english"
)
#cosine similarity finds similarities in 2 documents 
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)

#set Title column as an index, to find similar content by
#giving TV show or movie title as input 
indices = pd.Series(
    data.index, 
    index=data['cleanTitle']).drop_duplicates()
#print(indices)
#actual recommendation function:
def netFlix_recommendation(title, similarity = similarity):
    index = indices.get(clean(title))
    if index is None:
        return title + " is not in the database" 
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    movieindices = [i[0] for i in similarity_scores]
    return data['Title'].iloc[movieindices]

#print(netFlix_recommendation("realityhigh"))

print("Enter your movie or TV show title here:")
titleName = input()
print(netFlix_recommendation(titleName))
"""
"""
