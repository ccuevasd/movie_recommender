from flask import Flask, render_template
from recommender import random_recommend, MOVIES

app = Flask(__name__)
# __name__ is simply a reference to the current python script/module

@app.route('/')
# route of a website is called index
@app.route('/index')
def index():
    return render_template('index.html')
    #return      "<h1>Welcome to my website!</h1>"

# @app.route('/hello/<name>')
# def hello(name):
#     # in project: put preprocessing here
#     # in project NMF function
#     name = name.upper()
#     return render_template('hello.html', name_html=name)
#     # f"Hello, {name}  !"

@app.route('/recommendation')
def recommend():
    movies = random_recommend(MOVIES, 3)
    return render_template('recommendation.html', movies=movies)

if __name__ == '__main__':
    # if I run "python application.py", please run the following code
    app.run(debug=True, use_reloader=True)
