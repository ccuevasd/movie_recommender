import random
import pandas as pd

df = pd.read_csv(
    '/Users/Simon/Documents/Data_Science/Spiced_Academy/Course_Work/Week__10/Project/data/movies.csv')

MOVIES = df['title'].to_list()


def random_recommend(movies, num):
    random.shuffle(movies)
    return movies[:num]
