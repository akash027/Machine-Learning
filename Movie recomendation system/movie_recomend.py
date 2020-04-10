import numpy as np
import  pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#------------- function we will need them later -----------#

def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]


def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]

#-----------------------------------------------------------#
    

# read csv file

df = pd.read_csv("/home/sky/Documents/1.machine learning/Movie recomendation system/movie_dataset.csv")

print(df.head())

print(df.columns)


# Select Features

features = ['keywords', 'cast', 'genres', 'director']


# create a column in df which combines all selected features

for feature in features:
    df[feature] = df[feature].fillna('')
    

def combine_features(row):
    try:
        return row['keywords'] +" "+row['cast'] +" "+row['genres'] +" "+row['director']
    except:
        print("Error: ",row)


df["combined_features"] = df.apply(combine_features, axis=1)

print(df['combined_features'].head())



# create count matrix from this new combined column

cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])



#compute the cosine similarity based on count_matrix

cosine_sim = cosine_similarity(count_matrix)

print(cosine_sim.shape)


#Get index of the movie from its title
movie = 'Avatar'
movie_index = get_index_from_title(movie)

similar_movies = list(enumerate(cosine_sim[movie_index]))


#get list of similar movies in descending order of similarity score

sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)

print(sorted_similar_movies)


#get titles of some movies

i = 0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i += 1
    if i>20:
        break
















