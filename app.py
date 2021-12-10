from flask import Flask, render_template, request
from surprise import dump
import pandas as pd

app = Flask(__name__)

'''app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:Yinxy19980128@127.0.0.1:3306/movie_data"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "20211210"

db = SQLAlchemy(app)'''

model_tuple = dump.load("D:\py_movie_recommendation_system\EDAModel\svd")
model = model_tuple[1]


@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # Converting all the form values to float and making them append in a list(features)
        features = [float(x) for x in request.form.values()]
        n = int(features[0])
        uid = int(features[1])
        movies_df = pd.read_csv("D:/py_movie_recommendation_system/data/" + "movies.csv")
        ratings_df = pd.read_csv("D:/py_movie_recommendation_system/data/" + "ratings.csv").drop("timestamp", axis=1)
        # get all users
        users = list(ratings_df.userId.unique())
        # conner case if userId not in database
        if uid not in users:
            # find the top viewed movies as recommendation
            top = ratings_df[["movieId", "rating"]].groupby("movieId").count().reset_index().sort_values("rating",
                                                                                                         ascending=False).reset_index(
                drop=True).drop("rating", axis=1)
            all_top = pd.merge(top, movies_df, left_on="movieId", right_on="movieId").drop("movieId", axis=1)
            recommendation = all_top.head(n).reset_index(drop=True)
            recommendation_string = " | ".join(list(recommendation["title"]))
            return recommendation_string
        # get the recommendation using svd model
        movies_df["estimate_rating"] = movies_df["movieId"].apply(lambda x: model.predict(uid, x).est)
        user_recommendation = movies_df.drop("movieId", axis=1)
        user_recommendation = user_recommendation.sort_values("estimate_rating", ascending=False)
        recommendation = user_recommendation.head(n).reset_index(drop=True)
        recommendation_string = " | ".join(list(recommendation["title"]))
        return recommendation_string


# It is the starting point of code
if __name__ == '__main__':
    # We need to run the app to run the server
    app.run(debug=False)
