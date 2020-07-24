from flask import Flask, render_template, request
from recommender import random_recommend, MOVIES

app = Flask(__name__)
#__name__ is simply a reference to the current python script / module.


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', choices=MOVIES)


@app.route('/hello/<name>')
def hello(name):
    return render_template('hello.html', name_html=name)

@app.route('/recommendation')
def recommend():
    user_input = dict(request.args)

    """Here is where you take the user input dictionary and pre-process it
       for your recommendation function"""

    # movies = nmf_recommend(user_input)
    movies = random_recommend(MOVIES, 5)
    return render_template('recommendation.html', movies=movies)


if __name__ == '__main__':
    #if I run "python application.py", please run the following code....
    app.run(debug=True, use_reloader=True)
