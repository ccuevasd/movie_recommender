import random
import pandas as pd

df = pd.read_csv(
    '/Users/camilocuevas/SPICED/movie_recommender/data/movies.csv')


MOVIES = df['title'].to_list()


def random_recommend(movies, num):
    random.shuffle(movies)
    return movies[:num]
