import numpy as np
import pandas as pd
import json

# 데이터 불러오기 / 장르, 무비아이디, 타이틀
# https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

# pd.set_option('display.max_columns', None)
meta = pd.read_csv('./movies_metadata.csv', low_memory=False)
#meta.head()

meta = meta[['id', 'original_title', 'original_language', 'genres']]
meta = meta.rename(columns={'id':'movieId'})
meta = meta[meta['original_language'] == 'en']
meta.head()

ratings = pd.read_csv('./ratings_small.csv') # 평가데이터
ratings = ratings[['userId', 'movieId', 'rating']]
ratings.head()

# ratings.describe()

# 데이터 정제

meta.movieId = pd.to_numeric(meta.movieId, errors='coerce')
ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')

def parse_genres(genres_str):
    genres = json.loads(genres_str.replace('\'', '"'))

    genres_list = []
    for g in genres:
        genres_list.append(g['name'])

    return genres_list

meta['genres'] = meta['genres'].apply(parse_genres)

#meta.head()

# Merge Meta and Ratings
data = pd.merge(ratings, meta, on='movieId', how='inner') # 기준, 방식, 영화에 대한 평가를 나열
#data.head()

# Pivit Table
matrix = data.pivot_table(index='userId', columns='original_title', values='rating')
#matrix.head(20)

# Pearson Correlation
GENRE_WEIGHT = 0.1

def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))

def recommend(input_movie, matrix, n, similar_genre=True):
    input_genres = meta[meta['original_title'] == input_movie]['genres'].iloc(0)[0]

    result = []
    for title in matrix.columns:
        if title == input_movie:
            continue

        # rating comparison
        cor = pearsonR(matrix[input_movie], matrix[title])

        # genre comparison
        if similar_genre and len(input_genres) > 0:
            temp_genres = meta[meta['original_title'] == title]['genres'].iloc(0)[0]

            same_count = np.sum(np.isin(input_genres, temp_genres)) # 배열을 비교하여 똑같은 요소가 있으면 반환
            cor += (GENRE_WEIGHT * same_count)

        if np.isnan(cor):
            continue
        else:
            result.append((title, '{:.2f}'.format(cor), temp_genres))

    result.sort(key=lambda r: r[1], reverse=True) # 내림차순

    return result[:n]

# 예측
def recommend_movie(movie_name):
    print('추천 함수 동작', movie_name)
    recommend_result = recommend(movie_name, matrix, 10, similar_genre=True)
    pd.DataFrame(recommend_result, columns = ['Title', 'Correlation', 'Genre'])
    print(recommend_result, type(recommend_result))
    return recommend_result
