#!/usr/bin/env python
# coding: utf-8

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import KNNImputer

# Data
R = pd.read_csv('../data/UserRatingTitles-withoutYear.csv', index_col=0)

# KNN
imputer = KNNImputer(n_neighbors=5)
R = pd.DataFrame(imputer.fit_transform(R), columns=R.columns, index=R.index)

# cosine_matrix
similarities = pd.DataFrame(cosine_similarity(
    R), columns=R.index, index=R.index)


def convert_flask_dict(flask_dict):
    new_keys = list(flask_dict.values())[::2]
    new_vals = list(flask_dict.values())[1::2]

    return dict(zip(new_keys, new_vals))


# function working for user_input = {'Toy Story': 5, 'Sabrina': 1, 'My Man Godfrey': 2}
def get_recommendations_cosine(user_input):
    flask_user_input = user_input

    # Introduce New User
    new_user_df = pd.DataFrame(flask_user_input, index=['NewUser'])

    # Append newuser to the user-item matrix
    R_new_user = R.append(new_user_df)

    # Fill the NaNs for Paul
    R_new_user_filled = R_new_user.fillna(2.5)

    # Create a filter for the missing movies
    movie_filter = ~R_new_user.isna().any().values

    # Create an updated user list
    updated_users = R_new_user.index

    R_new_user.transpose()[movie_filter].transpose()

    # calculate a similarity to the other users based on dummys rating
    similarities_new_user = pd.DataFrame(cosine_similarity(R_new_user.transpose(
    )[movie_filter].transpose()), index=updated_users, columns=updated_users)

    # Predict ratings for Dummy
    similarities_new_user = similarities_new_user['NewUser'][~(
        similarities_new_user.index == 'NewUser')]

    # Calculate rating predictions
    rating_predictions = pd.DataFrame(
        np.dot(similarities_new_user, R)/similarities_new_user.sum(), index=R.columns)

    # Get recommendations
    #rating_predictions[~movie_filter].sort_values(by=0, ascending=False)
    films_recommended = rating_predictions[~movie_filter].sort_values(
        by=0, ascending=False).index[:3]

    return films_recommended
