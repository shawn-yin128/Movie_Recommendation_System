from comet_ml import Experiment

import pandas as pd
import numpy as np
from metaflow import FlowSpec, step, current
from datetime import datetime
import os

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


# get recommendation for user
def get_top_n(best_model, uid, n=10, location="/Users/yinxiangyang/desktop/code/final/data/"):
    """
    for user with uid, give him/her top n recommendations
    :param best_model: svd model
    :param uid: user id
    :param n: top n
    :param location: data path
    :return: dataframe of recommendation
    """
    # load data
    movies_df = pd.read_csv(location+"movies.csv")
    ratings_df = pd.read_csv(location+"ratings.csv").drop("timestamp", axis=1)
    # get all users
    users = list(ratings_df.userId.unique())
    # conner case if userId not in database
    if uid not in users:
        print(f"user {uid} is not in the database.")
        print("Here are the top reviewed movies:")
        # find the top viewed movies as recommendation
        top = ratings_df[["movieId", "rating"]].groupby("movieId").count().reset_index().sort_values("rating", ascending=False).reset_index(drop=True).drop("rating", axis=1)
        all_top = pd.merge(top, movies_df, left_on="movieId", right_on="movieId").drop("movieId", axis=1)
        recommendation = all_top.head(n).reset_index(drop=True)
        print(recommendation)
        return recommendation
    # get the recommendation using svd model
    movies_df["estimate_rating"] = movies_df["movieId"].apply(lambda x: best_model.predict(uid, x).est)
    user_recommendation = movies_df.drop("movieId", axis=1)
    user_recommendation = user_recommendation.sort_values("estimate_rating", ascending=False)
    recommendation = user_recommendation.head(n).reset_index(drop=True)
    print(recommendation)
    return recommendation


# for movie
def recommend_similar(mid, n=10, location="/Users/yinxiangyang/desktop/code/final/data/"):
    """
    find similar movies given a movie id
    :param mid: movie id
    :param n: n similar movies
    :param location: data path
    :return: dataframe of similar movies
    """
    # load datas
    movies_df = pd.read_csv(location+"movies.csv")
    ratings_df = pd.read_csv(location+"ratings.csv").drop("timestamp", axis=1)
    movies_list = list(movies_df["movieId"])
    # if this movie is not in database, we cannot make predictions
    if mid not in movies_list:
        print("This is a new movie and we cannot find similar movie only based on it id.")
        return
    # for movie in database, we calculate a cosine similarity
    print(f"Top {n} movies recommended based on cosine similarity")
    # split genres and encoding with one-hot encoding
    movies_df_encoding = pd.concat([movies_df[["movieId", "title"]], movies_df["genres"].str.get_dummies(sep='|').astype(np.int64)], axis=1)
    target_movie = movies_df_encoding[movies_df_encoding["movieId"]==mid].reset_index(drop=True)
    other_movie = movies_df_encoding[movies_df_encoding["movieId"]!=mid].reset_index(drop=True)
    # calculate cosine similarity
    from scipy import spatial
    cosine_result = []
    for x in range(other_movie.shape[0]):
        other_movie_list = list(other_movie.iloc[x, 2:].apply(int))
        target_movie_list = list(target_movie.iloc[0:, 2:].apply(int))
        cosine = spatial.distance.cosine(other_movie_list, target_movie_list)
        cosine_result.append(cosine)
    result_df = pd.DataFrame(cosine_result)
    result_df.columns = ["Cosine"]
    similar_df = pd.concat([other_movie, result_df], axis=1)
    similar = pd.merge(similar_df[["movieId", "title", "Cosine"]], movies_df[["movieId", "genres"]], left_on="movieId", right_on="movieId").sort_values("Cosine", ascending=False).drop("Cosine", axis=1).head(n).reset_index(drop=True)
    print(similar)
    return similar


class MovieRecommendation(FlowSpec):
    @step
    def start(self):
        """
        start up and print some info
        """
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)

        self.next(self.load_data)

    @step
    def load_data(self):
        """
        read data from file using pyspark
        """
        # set up data location and load data
        location = "/Users/yinxiangyang/desktop/code/final/data/"
        self.movies_df = pd.read_csv(location + "movies.csv")
        self.ratings_df = pd.read_csv(location + "ratings.csv")
        self.links_df = pd.read_csv(location + "links.csv")
        self.tags_df = pd.read_csv(location + "tags.csv")

        # next step is check if the dataset is valid
        self.next(self.check_data)

    @step
    def check_data(self):
        """
        make sure there is data inside the dataset
        """
        # a very basic and valid dataset should have data
        assert self.movies_df.shape[0] != 0
        assert self.ratings_df.shape[0] != 0
        assert self.links_df.shape[0] != 0
        assert self.tags_df.shape[0] != 0

        # next step is to run some basic preprocessing on the dataset
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        """
        preprocess data, includes followings steps:
            1. drop useless column and na
            2. datatype convert
        """
        # drop na
        movies_df_dropna = self.ratings_df.dropna()

        # drop timestamp which is not useful for this model
        movies_df_drop = movies_df_dropna.drop("timestamp", axis=1)

        # convert string to right type
        user_movie = movies_df_drop[["userId", "movieId"]].astype(int)
        rating = movies_df_drop["rating"].astype(float)
        self.movie_ratings = pd.concat([user_movie, rating], axis=1)

        # next step is to split the dataset into train and test
        self.next(self.train_test_split)

    @step
    def train_test_split(self):
        """
        use pyspark dataframe built-in api randomSplit to splie dataset into train and test
        """
        # train test split, here we only have train and test, no x and y because its non-supervised
        # we use 80% as train and 20% as test
        from surprise import Reader, Dataset
        from surprise.model_selection import train_test_split
        reader = Reader()
        self.data = Dataset.load_from_df(self.movie_ratings[["userId", "movieId", "rating"]], reader)
        self.train, self.test = train_test_split(self.data, test_size=0.2)

        # next step is to use grid search and cv to get a best model
        self.next(self.model_selection)

    @step
    def model_selection(self):
        """
        define a grid search and cv method to do model selection and pick the best model
        """
        from surprise import SVD
        from surprise.model_selection import GridSearchCV
        # define grid
        param_grid = {'n_factors': [25, 50, 100], 'n_epochs': [10, 20, 30], 'lr_all': [0.001, 0.005, 0.01],
                      'reg_all': [0.005, 0.001, 0.05]}

        # define grid search cv and use the whole set to select
        gscv = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=5)
        gscv.fit(self.data)

        # get the best model
        params = gscv.best_params['rmse']
        self.svd = SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'], lr_all=params['lr_all'],
                  reg_all=params['reg_all'])

        # upload to comet
        best_params = {
            "n_factors": params["n_factors"],
            "n_epochs": params["n_epochs"],
            "lr_all": params["lr_all"],
            "reg_all": params["reg_all"]
        }
        exp = Experiment("h7514oDijzJ9dQDKM44SxZ8Wi", project_name="LearningMachineFinalProject")
        exp.log_parameters(best_params)
        exp.log_metric("RMSE", gscv.best_score["rmse"])

        # next step it to set up a cross validation to tune model
        self.next(self.quantitative_test)

    @step
    def quantitative_test(self):
        """
        test the best model
        """
        # test best model with rmse
        from surprise import accuracy
        predictions = self.svd.fit(self.train).test(self.test)
        accuracy.rmse(predictions, verbose=True)

        # next step is to do some qualitative check
        self.next(self.qualitative_check)

    @step
    def qualitative_check(self):
        """
        this part contains two aspects:
            1. make recommendation to a user
            2. find similar movie to an input movie
        """
        # for user
        ## for user in dataset
        out1 = get_top_n(self.svd, 1, 10)

        ## for user not in dataset
        out2 = get_top_n(self.svd, 1111111111, 10)

        # for movie
        ## in dataset
        out3 = recommend_similar(858, 10)

        ## not in dataset
        out4 = recommend_similar(1111111111, 10)

        # all works done, to the end
        self.next(self.end)

    @step
    def end(self):
        """
        this is the end of flow
        """
        print("All done at {}!".format(datetime.utcnow()))


# main funciton
if __name__ == "__main__":
    MovieRecommendation()
