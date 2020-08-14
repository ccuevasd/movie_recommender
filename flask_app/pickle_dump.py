#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer
from fuzzywuzzy import process
import pickle

# load data
R = pd.read_csv('../data/UserRatingTitles-withoutYear.csv', index_col=0)

# impute missing values with KNN
imputer = KNNImputer(n_neighbors=5)
R = pd.DataFrame(imputer.fit_transform(R), columns=R.columns, index=R.index)

# train model
model = NMF(19)
model.fit(R)

# Export inputed R and pretrained model
pickle.dump(R, open('R_binary', 'wb'))
pickle.dump(model, open('nmf_binary', 'wb'))
