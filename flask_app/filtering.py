# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


df_MovGen = pd.read_csv(r'/Users/Simon/Documents/Data_Science/Spiced_Academy/Course_Work/Week__10/Project/data/MovGenRec.csv')

def recommend_by_genre(UserInput):
    recommendations = df_MovGen[df_MovGen[UserInput] == 1].sort_values(by='Num_Ratings', ascending=False)[:50].sort_values(by='Mean_Rating', ascending=False)[:10]['title']
    return list(recommendations)

print(recommend_by_genre('Comedy'))
