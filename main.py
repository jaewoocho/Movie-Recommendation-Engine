import numpy as np
import pandas as pd
import json

# Loading Movie csv Dataset 
meta = pd.read_csv('movies_metadata.csv')
meta.head()

# Preprocess data 
meta = meta[['id', 'original_title', 'original_language', 'genres']]
meta = meta.rename(columns={'id' : 'movieId'})
meta = meta[meta['original_language'] == 'en']
meta.head()

ratings = pd.read_csv('ratings_small.csv')
ratings = ratings[['userId', 'movieId', 'rating']]
ratings.head()

ratings.describe()

# Refine/Refactor Dataset 
meta.movieId = pd.to_numeric(meta.movieId, errors='coerce')
ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')

def parse_genres(genres_str):
  genres = json.loads(genres_str.replace('\', '""'))
  genres_list = []
  for g in genres: 
    genres_list.append(g['name'])                                         
  return genres_list
  
meta['genres'] = meta['genres'].apply(parse_genres)
meta.head()
                                         
# Merge Metadata and Ratings 
data = pd.merge(ratings, meta, on = 'movieId', how='inner')
data.head()

# Pivot Table 
matrix = data.pivot_table(Index='userId', columns='original_title', values = 'rating')
matrix.head(20)   

                                         
                                         
# Pearson Correlation
# https://statistics.laerd.com/statistical-guides/pearson-correlation-coefficient-statistical-guide.php
                                         
GENRE_WEIGHT = 0.1

def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c /  np.sqrt(np.sum(s1_c **2) * np.sum(s2_c ** 2))

def recommend(input_movie, matrix, n, similar_genre=True):
    input_genres = meta[meta['original_title'] == input_movie]['genres'].lloc(0)[0]                  

    result  = []
    for title in matrix.columns:
      if title == input_movie:
          continue          

    # rating comparison
    cor = pearsonR(matrix[input_movie], matrix[title])

    # genre comparison
    if similar_genre and len(input_genres) >0:
      temp_genres = meta[meta['original_title'] == title]['genres'].iloc(0)[0]
      same_count = np.sum(np.isin(input_genres, temp_genres))                  
      cor += (GENRE_WEIGHT * same_count)                  

    if np.isnan(cor):
        continue
    else:
        result.append((title, ' {:.2f}'.format(cor), temp_genres))
                  
result.sort(key=lambda r: r[1], reverse=True)                  
                  
return result[:n]                  
                     
# Prediction                  
recommend_result = recommend('The Dark Knight', matrix, 10, similar_genre=True)                  

pd.DataFrame(recommend_result, columns = ['Title', 'Correlation', 'Genre'])                  
                                         

                                         

                                         
                                         
                                         
                                         
                                         
                                        
