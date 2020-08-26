# movie_recommender

This movie recommender has been built using unsupervised learning. You can choose between 2 different algorithms to get a recommendation, or alternatively you can choose the genre of movie your would like to watch. The Non-Negative Matrix Factorization model requires the user to rate 3 movies, this together with decomposing data found on the MovieLens data sets, we are able to extract useful freatures and generate recommendations. The Cosine Similarity model also requires the user to rate 3 movies and then numerically, it gauges the cosine of the edge between two vectors anticipated in a multi-dimensional space.

The application uses flask, we have improved the performance by storing images of previosly generated python objects and we have used KNN Imputer to fill in the missing values. 

