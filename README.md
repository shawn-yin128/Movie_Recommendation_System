# Movie Recommendation System
This is a final project for nyu tandon fre 7773 Machine Learning course. The goal is to build a movie recommendation system and deploy using flask.
## Motivation:
Recommendation system is a very basic and widely used system in production. When you watch videos, there is recommendation system. When you shop online, there is recommendation system.

So, we choose this topic to make myself stand out in the future when searching for a job and meanwhile help me to adopt more productions.
## Two Projects:
### In this projects, there are actually two projects. 

#### The first one uses Alternating Least Squares (ALS) with API in PySpark. You can find the notebook file in Deprecated directory.
This project works well in notebook, but the problem is the model cannot be saved using pickle package. Meanwhile, Hadoop works with tons of errors on my computer, which makes it harder for deployment, as a result, we decide to drop this project.

#### The second project uses Singular Value Decomposition (SVD) with API in scikit-surprise package. You can find it in EDAModel directory.
SVD is another algorithm for building a recommendation system, and surprise package has SVD API. Meanwhile, we still use pyspark to do the data exploration part and for the model part we use surprise to do most of the works. We also use flask to delpoy this model locally.
