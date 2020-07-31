#!/usr/bin/env python
# coding: utf-8

# Libraries
import pickle
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF
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
    return_dict = dict(zip(new_keys, new_vals))
    return_dict['method']= flask_dict['method']
    return return_dict


# function working for user_input = {'Toy Story': 5, 'Sabrina': 1, 'My Man Godfrey': 2}
def get_recommendations_cosine(user_input):
    flask_user_input = dict(list(user_input.items())[:-1])

    # Introduce New User
    #new_user_df = pd.DataFrame(flask_user_input, index=['NewUser'])
    new_user_df = pd.DataFrame(
        [np.nan]*len(R.columns), index=R.columns)
    new_user_df.columns = ['NewUser']
    new_user_df = new_user_df.transpose()

    # Loop to check whether user_id is there
    for key, value in flask_user_input.items():
        if key in new_user_df.columns:
            new_user_df.loc[:, key] = float(value)
        else:
            closest_match = process.extract(key, R.columns)[0][0]
            new_user_df.loc[:, closest_match] = float(value)
            if len(process.extract(key, R.columns)[0][0]) < 0.5*len(key):
                closest_match = process.extract(key, R.columns)[1][0]
                new_user_df.loc[:, closest_match] = float(value)

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


def get_recommendations_nmf(user_input):
    flask_user_input = dict(list(user_input.items())[:-1])
    # flask_user_input = {
    # 'toy Story': '5',
    #  'Jumanyi': '3',
    #  'Grupmyer Old Men': '4'
    #  }

    new_user_vector = pd.DataFrame(
        [np.nan]*len(R.columns), index=R.columns).transpose()

    # Loop to check whether user_id is there
    for key, value in flask_user_input.items():
        if key in new_user_vector.columns:
            new_user_vector.loc[:, key] = float(value)
        else:
            closest_match = process.extract(key, R.columns)[0][0]
            new_user_vector.loc[:, closest_match] = float(value)
            if len(process.extract(key, R.columns)[0][0]) < 0.5*len(key):
                closest_match = process.extract(key, R.columns)[1][0]
                new_user_vector.loc[:, closest_match] = float(value)
                # print(closest_match)

    # Fill in the missing values
    new_user_vector_filled = new_user_vector.fillna(2.5)

    # Calculate the hidden profile with nmf.transform # user-feature_matrix of new user
    hidden_profile = model.transform(new_user_vector_filled)

    # Calculate the predictions using np.dot
    rating_prediction = pd.DataFrame(
        np.dot(hidden_profile, model.components_), columns=new_user_vector.columns)

    # Create a boolean mask to filter out the positions where the data was originally NaN
    bool_mask = np.isnan(new_user_vector.values[0])

    # Find the movies that have not yet been seen
    movies_not_seen = rating_prediction.columns[bool_mask]

    # Find recommendations for unseen movies
    movies_not_seen_df = rating_prediction[movies_not_seen].T

    # Get recommendations
    films_recommended = movies_not_seen_df.sort_values(
        by=0, ascending=False).index[:3]
    return films_recommended
# print(get_recommendations(user_input))
