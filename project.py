# %%
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd

# %%
import argparse
parser = argparse.ArgumentParser(
    description='Recommender system using NMF, SVD or Stochastic Gradient Descent.')
parser.add_argument(
    '-tr', '--train', type=str, metavar='',
    required=True, help='Path to trainfile with movie ratings.')
parser.add_argument(
    '-ts', '--test', type=str, metavar='',
    required=True, help='Path to testfile with movie ratings.')
parser.add_argument(
    '-a', '--alg', type=str, metavar='', choices=['SVD1', 'SVD2', 'NMF', 'SGD'],
    required=True, help='Chosen algorithm for Recommender system (NMF, SVD1, SVD2, SGD)')
parser.add_argument(
    '-r', '--result', type=str, metavar='', required=True, help='Path where root-mean square error (RMSE) is to be saved.')
args = parser.parse_args()

# %%
train = pd.read_csv(filepath_or_buffer=args.train)
test = pd.read_csv(filepath_or_buffer=args.test)

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
    return np.sqrt(s/len(not_nans))


# %%
def nmf(matrix, r, max_iter=1000):
    model = NMF(n_components=r, init='random',
                random_state=0, max_iter=max_iter)
    W = model.fit_transform(matrix)
    H = model.components_
    Z_approximated = np.dot(W, H)
    return Z_approximated


# %%
def svd_1(Z: np.ndarray, r: int = 7) -> np.ndarray:
    """
    Function performs low rank approximation of utility matrix Z via truncated SVD.

    :param Z: Utility matrix Z (users x movies)
    :type Z: numpy.ndarray
    :param r: Number of components in truncated SVD (ranks)
    :type r: int
    :return: Rank r matrix, obtained from the truncated SVD being LR approximation of Z.
    :rtype: numpy.ndarray
    """
    svd = TruncatedSVD(n_components=r, random_state=42)
    svd.fit(Z)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(Z)/svd.singular_values_
    H = np.dot(Sigma2, VT)
    Z_approximated = np.dot(W, H)
    return Z_approximated


def svd_2(Z: np.ndarray, r: int, Z_test: np.ndarray, n_iter: int = 100, update: int = 10) -> np.ndarray:
    """Function performs iterative low rank approximation of utility matrix Z via truncated SVD.

    :param Z: Train utility matrix Z (users x movies)
    :type Z: numpy.ndarray
    :param r: Number of components in truncated SVD (ranks)
    :type r: int
    :param Z_test: Test utility matrix Z (users x movies)
    :type Z_test: numpy.ndarray
    :param n_iter: Number of iterations, defaults to 100
    :type n_iter: int, optional
    :param update: Print RMSE of approximation every x iterations, defaults to 10
    :type update: int, optional
    :return: Rank r matrix, obtained from the iterative truncated SVD being LR approximation of Z.
    :rtype: numpy.ndarray
    """
    Z = fill_zeros(Z)
    Zr = fill_zeros(Z)

# with tqdm(total=100) as pbar:
    for i in range(n_iter):
        Z_approximated = svd_1(Zr, r)
        Zr = (Z == 0).astype(float) * Z_approximated + Z
        if i % update == 0:
            print(f'RMSE after {i}th iteration equals = {RMSE(Zr, Z_test)}')
        # pbar.update(10)
    return Zr


def fill_zeros(matrix):
    return np.nan_to_num(matrix, nan=0.0)


def fill_mean_global(matrix):
    m = np.nanmean(matrix)
    return np.nan_to_num(matrix, nan=m)


def fill_mean_movies(matrix):
    col_mean = np.nanmean(matrix, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    inds = np.where(np.isnan(matrix))
    matrix_copy = matrix.copy()
    matrix_copy[inds] = np.take(col_mean, inds[1])
    return matrix_copy


def fill_mean_users(matrix):
    row_mean = np.nanmean(matrix, axis=1)
    row_mean = np.nan_to_num(row_mean, nan=0.0)
    inds = np.where(np.isnan(matrix))
    matrix_copy = matrix.copy()
    matrix_copy[inds] = np.take(row_mean, inds[0])
    return matrix_copy


# %%
# Experiment 1
filled_train_ratings = fill_mean_users(train_ratings)

if args.alg == 'NMF':
    approximation = nmf(filled_ratings, 6, max_iter=1000)
elif args.alg == 'SVD1':
    approximation = svd_1(filled_ratings, 6)
elif args.alg == 'SVD2':
    approximation = svd_2(Z=train_ratings, r=7,
                          Z_test=test_ratings, n_iter=100, update=10)
elif args.alg == 'SGD':
    pass

rmse_test = RMSE(approximation, test_ratings)

with open(args.result, 'a') as f:
    f.write(str(rmse_test))
