from flask import Flask, render_template, request
from recommender import random_recommend, MOVIES
from models import get_recommendations_cosine, get_recommendations_nmf, convert_flask_dict
# from Cosine import get_recommendations_cosine #convert_flask_dict

app = Flask(__name__)
# __name__ is simply a reference to the current python script/module


@app.route('/')
@app.route('/index')  # route of a website is called index
def index():
    return render_template('index.html', choices=MOVIES)
    # return      "<h1>Welcome to my website!</h1>"


@app.route("/recommendation")
def recommend():
    user_input = dict(request.args)
    user_input = convert_flask_dict(user_input)

    method_ = user_input['method']
    if method_ == "NMF":
        movies = get_recommendations_nmf(user_input)
    if method_ == "Cosine":
        movies = get_recommendations_cosine(user_input)
    return render_template('recommendation.html', movies=movies)


if __name__ == '__main__':
    # if I run "python application.py", please run the following code
    app.run(debug=True, use_reloader=True)
