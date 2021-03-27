# %%
import numpy as np
import pandas as pd
# %%
train = pd.read_csv('train_ratings.csv')
test = pd.read_csv('test_ratings.csv')

users = set(train.userId)
train_movies = set(train.movieId)
test_movies = set(test.movieId)
movies = train_movies.union(test_movies)
# movies_not_in_test = list(movies.difference(test_movies))
# movies_not_in_train = list(movies.difference(train_movies))
users = list(users)
movies = list(movies)
user_ids = dict((index, i) for (i, index) in enumerate(users))
movie_ids = dict((index, i) for (i, index) in enumerate(movies))
# %%
train_ratings = np.zeros((len(users), len(movies)))
train_ratings[:] = np.nan

for row in train.itertuples():
  train_ratings[user_ids[row.userId], movie_ids[row.movieId]] = row.rating


test_ratings = np.zeros((len(users), len(movies)))
test_ratings[:] = np.nan

for row in test.itertuples():
  test_ratings[user_ids[row.userId], movie_ids[row.movieId]] = row.rating
# %%
def RMSE(prediction, truth):
  not_nans = np.argwhere(~np.isnan(truth))
  s = 0
  for (user_id, movie_id) in not_nans:
    s += (prediction[user_id, movie_id]-truth[user_id, movie_id])**2
  return s/len(not_nans)

# %%
# %%
