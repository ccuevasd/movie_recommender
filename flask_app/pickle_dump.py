#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer
from fuzzywuzzy import process
import pickle


R = pd.read_csv('../data/UserRatingTitles-withoutYear.csv', index_col=0)

imputer = KNNImputer(n_neighbors=5)
R = pd.DataFrame(imputer.fit_transform(R), columns=R.columns, index=R.index)

model = NMF(19)
model.fit(R)

Q = pd.DataFrame(model.components_, columns=R.columns)

P = pd.DataFrame(model.transform(R), index=R.index)

pickle.dump(nmf, open('nmf_binary', 'wb'))
