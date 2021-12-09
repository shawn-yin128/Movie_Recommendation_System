from comet_ml import Experiment

import pandas as pd
import numpy as np
from metaflow import FlowSpec, step, current
from datetime import datetime
import os

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


# define a function to package the recommendation
def top_K_recommend(k, uid, model, data_location="D:/py_movie_recommendation_system/data/"):
    """
    k: the number of movies to recommend
    id: the id of the user to give recommendations
    model: the trained model for recommendation
    """
    # the table for all top10 recommendations
    from pyspark.sql import SparkSession
    spark = SparkSession \
        .builder \
        .appName("moive analysis") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    movies_df = spark.read.load(data_location + "movies.csv", format='csv', header=True)

    all_recommend = model.recommendForAllUsers(k)
    user_recommend = all_recommend.where(all_recommend.userId == uid).toPandas()
    if user_recommend.shape[0] == 0:
        print('No user with id ' + str(uid) + ' is found in the data.')
        print("Would you like to watch most frequently watched movie?")
        ratings_df = spark.read.load(data_location + "ratings.csv", format='csv', header=True)
        movies_df.registerTempTable("movies")
        ratings_df.registerTempTable("ratings")
        out = spark.sql(
            f"""
            SELECT a.movieId, a.title, a.genres
            FROM movies AS a
            LEFT JOIN (
            SELECT movieId, COUNT(rating) AS rating
            FROM ratings
            GROUP BY movieId
            ) AS b
            USING(movieId)
            ORDER BY b.rating DESC
            LIMIT {k}
            """
        ).toPandas()
        return out
    user_recommend = user_recommend.iloc[0, 1]
    user_recommend = pd.DataFrame(user_recommend, columns=['movieId', 'predicted_ratings'])
    temp = None
    for i in user_recommend['movieId']:
        if not temp:
            temp = movies_df.where(movies_df.movieId == str(i))
        else:
            temp = temp.union(movies_df.where(movies_df.movieId == str(i)))
    out = pd.concat([temp.toPandas(), user_recommend['predicted_ratings']], axis=1)
    out.index = range(1, k + 1)
    return out


# cosine similarity
# the larger the cosine value, the smaller the two feature vectors' angle, the similar the movies
# this similarity considers the direction only,
# e.g. movie 1 with factor [1,2,3] and movie 2 with factor [2,4,6] are considered the same
def cos_similar(k, mid, best_model, data_location="D:/py_movie_recommendation_system/data/"):
    """
    k: number of similar movies to find
    mid: id of the movie to find similarities
    """
    from pyspark.sql import SparkSession
    spark = SparkSession \
        .builder \
        .appName("moive analysis") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    movies_df = spark.read.load(data_location + "movies.csv", format='csv', header=True)
    movies_df.registerTempTable("movies")

    print("We are looking for similar movie of ", spark.sql(
        f"""
        SELECT title
        FROM movies
        WHERE movieId = {mid}
        """
    ).toPandas().iloc[0, 0])

    movie_factors = best_model.itemFactors
    movie_factors.printSchema()
    comd = ["movie_factors.selectExpr('id as movieId',"]
    for i in range(best_model.rank):
        if i < best_model.rank - 1:
            comd.append("'features[" + str(i) + "] as feature" + str(i) + "',")
        else:
            comd.append("'features[" + str(i) + "] as feature" + str(i) + "'")
    comd.append(')')
    movie_factors = eval(''.join(comd))
    movie_factors.createOrReplaceTempView('movie_factors')

    movie_info = spark.sql(
        f"""
        SELECT * 
        FROM movie_factors 
        WHERE movieId= {mid}
        """
        ).toPandas()
    if movie_info.shape[0] <= 0:
        print('No movie with id ' + str(mid) + ' is found in the data.')
        return None, None
    norm_m = sum(movie_info.iloc[0, 1:].values ** 2) ** 0.5
    temp = ['select movieId,']
    norm_str = ['sqrt(']
    for i in range(best_model.rank):
        comd = 'feature' + str(i) + '*' + str(movie_info.iloc[0, i + 1])
        temp.append(comd + ' as inner' + str(i) + ',')
        if i < best_model.rank - 1:
            norm_str.append('feature' + str(i) + '*feature' + str(i) + '+')
        else:
            norm_str.append('feature' + str(i) + '*feature' + str(i))
    norm_str.append(') as norm')
    temp.append(''.join(norm_str))
    temp.append(' from movie_factors where movieId!=' + str(mid))
    inner = spark.sql(' '.join(temp))
    inner = inner.selectExpr('movieId',
                             '(inner0+inner1+inner2+inner3+inner4)/norm/' + str(norm_m) + ' as innerP').orderBy(
        'innerP', ascending=False).limit(k).toPandas()
    out = None
    for i in inner['movieId']:
        if not out:
            out = movies_df.where(movies_df.movieId == str(i))
        else:
            out = out.union(movies_df.where(movies_df.movieId == str(i)))
    out = out.toPandas()
    out.index = range(1, k + 1)
    return out, inner


# write a function to make prediction for movie id
def similar_movie(k, mid, best_model, data_location="D:/py_movie_recommendation_system/data/"):
    out, inner = cos_similar(k, mid, best_model, data_location)
    print(out)
    if out is None:
        print("This is a new movie, and we cannot find similar movie only based on it id!")
    return out


# write a function to make recommendation
def make_recommendation(k, uid, best_model, data_location="D:/py_movie_recommendation_system/data/"):
    out = top_K_recommend(k, uid, best_model, data_location)
    print(out)
    return out


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
        # initiate spark and create interface
        from pyspark.sql import SparkSession
        self.spark = SparkSession \
            .builder \
            .appName("moive analysis") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

        # set up data location and load data
        location = "D:/py_movie_recommendation_system/data/"
        self.movies_df = self.spark.read.load(location + "movies.csv", format='csv', header=True)

        # next step is check if the dataset is valid
        self.next(self.check_data)

    @step
    def check_data(self):
        """
        make sure there is data inside the dataset
        """
        # a very basic and valid dataset should have data
        assert self.movies_df.count() != 0

        # take a look about what we have in the dataset
        print(self.movies_df.decribe())

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
        movies_df_dropna = self.movies_df.dropna()

        # drop timestamp which is not useful for this model
        movies_df_drop = movies_df_dropna.drop("timestamp")

        # convert string to right type
        from pyspark.sql.types import IntegerType, FloatType
        self.movie_ratings = movies_df_drop.withColumn("userId", movies_df_drop["userId"].cast(IntegerType()))
        self.movie_ratings = movies_df_drop.withColumn("movieId", movies_df_drop["movieId"].cast(IntegerType()))
        self.movie_ratings = movies_df_drop.withColumn("rating", movies_df_drop["rating"].cast(FloatType()))

        # have an overview about features and type
        print(self.movie_ratings.describe())

        # next step is to split the dataset into train and test
        self.next(self.train_test_split)

    @step
    def train_test_split(self):
        """
        use pyspark dataframe built-in api randomSplit to splie dataset into train and test
        """
        # train test split, here we only have train and test, no x and y because its non-supervised
        # we use 80% as train and 20% as test
        self.train, self.test = self.movie_ratings.randomSplit([0.8, 0.2])

        # next step is define the model and evaluator we gonna use
        self.next(self.model_define)

    @step
    def model_define(self):
        """
        define the model and evaluation method
        """
        from pyspark.ml.evaluation import RegressionEvaluator
        from pyspark.ml.recommendation import ALS
        # define model
        self.raw_model = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", seed=2021)

        # define evaluator and use RMSE as the evaluation metrics
        self.evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

        # next step it to set up a cross validation to tune model
        self.next(self.cross_validation)

    @step
    def cross_validation(self):
        """
        use a grid search to tune model
        """
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
        # build a grid for parameter
        self.params = ParamGridBuilder().addGrid(self.raw_model.maxIter, [3, 5, 10]).addGrid(self.raw_model.regParam, [0.1, 0.01, 0.001]).addGrid(self.raw_model.rank, [5, 10, 15]).addGrid(self.raw_model.alpha, [0.1, 0.001, 0.0001]).build()

        # build 5 fold cross validation
        self.cv = CrossValidator(estimator=self.raw_model, estimatorParamMaps=self.params, evaluator=self.evaluator, numFolds=5, seed=2021)

        # next step is to fit the cv model and find the best ALS model
        self.next(self.find_best_model)

    @step
    def find_best_model(self):
        """
        fit the cv and find the best model
        """
        # fit cv
        self.cv_model = self.cv.fit(self.train)

        # get best model
        self.best_model = self.cv_model.bestModel

        # next step is to have a quantitative test
        self.next(self.quantitative_test)

    @step
    def quantitative_test(self):
        """
        test the best model, and upload metric and parameters into comet_ml
        """
        # get the best model's parameters
        best_params = self.cv_model.getEstimatorParamMaps()[np.argmin(self.cv_model.avgMetrics)]

        # print out the best model parameters
        print('Best ALS model parameters by CV:')
        for i, j in best_params.items():
            print('-> ' + i.name + ': ' + str(j))

        # get predictions
        prediction_test = self.best_model.transform(self.test)

        # evaluate
        rmse_score = self.evaluator.evaluate(prediction_test)
        print("Root-mean-square error for testing data is " + str(rmse_score))

        # upload
        exp = Experiment("h7514oDijzJ9dQDKM44SxZ8Wi", project_name="LearningMachineFinalProject")
        exp.log_parameters(best_params)
        exp.log_metric("RMSE", rmse_score)

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
        out1 = make_recommendation(10, 500, self.best_model)

        ## for user not in dataset
        out2 = make_recommendation(10, 11111111111111111, self.best_model)

        # for movie
        ## in dataset
        out3 = similar_movie(10, 858, self.best_model)

        ## not in dataset
        out4 = similar_movie(10, 11111111111111111, self.best_model)

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