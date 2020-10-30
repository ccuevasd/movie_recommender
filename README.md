# movie_recommender

This movie recommender has been built using unsupervised learning. You can choose between 2 different algorithms to get a recommendation, or alternatively you can choose the genre of movie your would like to watch. The Non-Negative Matrix Factorization model requires the user to rate 3 movies, this together with decomposing data found on the MovieLens data sets, we are able to extract useful freatures and generate recommendations. The Cosine Similarity model also requires the user to rate 3 movies and then numerically, it gauges the cosine of the edge between two vectors anticipated in a multi-dimensional space.

The application uses flask, we have improved the performance by storing images of previosly generated python objects and we have used KNN Imputer to fill in the missing values. 

To run the code open the application.py from the flask_app folder and then visit the website running on the shown host.

![Screenshot 2020-10-30 at 16 16 33](https://user-images.githubusercontent.com/64790033/97723315-f1106980-1acb-11eb-9811-123fe40dcd56.png)
