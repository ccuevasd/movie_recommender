#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from fuzzywuzzy import process


nmf = pickle.load(open('nmf_binary', 'rb'))


def convert_flask_dict(flask_dict):
    new_keys = list(flask_dict.values())[::2]
    new_vals = list(flask_dict.values())[1::2]

    return dict(zip(new_keys, new_vals))


def get_recommendations(user_input):
    flask_user_input = user_input
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


q
# Get recommendations
films_recommended = movies_not_seen_df.sort_values(
    by=0, ascending=False).index[:3]
return films_recommended
# print(get_recommendations(user_input))
