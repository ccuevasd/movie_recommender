import random
import pandas as pd

df = pd.read_csv('../data/movies.csv', index_col=0)

MOVIES = df['title'].to_list()


def random_recommend(movies, num):
    random.shuffle(movies)
    return movies[:num]
