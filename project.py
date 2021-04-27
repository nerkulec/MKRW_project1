# %%
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

train_ratings, test_ratings = None, None


def get_matrices(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

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

    train_ratings = np.zeros((len(users), len(movies)))
    train_ratings[:] = np.nan
    for row in train.itertuples():
        train_ratings[user_ids[row.userId],
                      movie_ids[row.movieId]] = row.rating

    test_ratings = np.zeros((len(users), len(movies)))
    test_ratings[:] = np.nan
    for row in test.itertuples():
        test_ratings[user_ids[row.userId], movie_ids[row.movieId]] = row.rating
    return train_ratings, test_ratings

# %%


def RMSE(prediction, truth):
    not_nans = np.argwhere(~np.isnan(truth))
    s = 0
    for (user_id, movie_id) in not_nans:
        s += (prediction[user_id, movie_id]-truth[user_id, movie_id])**2
    return np.sqrt(s/len(not_nans))


# %%
def nmf(Z: np.ndarray, r: int = 7, max_iter: int = 1000) -> np.ndarray:
    """
    Function performs low rank approximation of utility matrix Z via Non-negative Matrix Factorization.

    :param Z: Utility matrix Z (users x movies)
    :type Z: numpy.ndarray
    :param r: Number of components in NMF (ranks)
    :type r: int
    :param max_iter: Maximum number of NMF algorithm iterations
    :type max_iter: int
    :return: Rank r matrix - an approximation of Z.
    :rtype: numpy.ndarray
    """
    model = NMF(n_components=r, init='random',
                random_state=0, max_iter=max_iter)
    W = model.fit_transform(Z)
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


def svd_2(
        Z_filled: np.ndarray, Z: np.ndarray, Z_test: np.ndarray, r: int, n_iter: int = 100,
        epsilon: float = 0.00001, verbose: bool = True) -> np.ndarray:
    """Function performs iterative low rank approximation of utility matrix Z via truncated SVD.

    :param Z_filled: Filled Train utility matrix Z (users x movies)
    :type Z_filled: numpy.ndarray
    :param Z: Train utility matrix Z (users x movies) before filling
    :type Z: numpy.ndarray
    :param Z_test: Test utility matrix Z (users x movies) to control algorithm convergence.
    :type Z_test: numpy.ndarray
    :param r: Number of components in truncated SVD (ranks)
    :type r: int
    :param n_iter: Number of iterations, defaults to 100
    :type n_iter: int, optional
    :param epsilon: Tolerance checking if algorithm is converging
    :type epsilon: float, optional
    :return: Rank r matrix, obtained from the iterative truncated SVD being LR approximation of Z.
    :rtype: numpy.ndarray
    """
    Z_0 = np.nan_to_num(Z, nan=0.0)
    Zr = Z_filled
    i = 0
    rmse_prev = np.Inf
    condition = True
    while condition:
        Z_approximated = svd_1(Zr, r)
        Zr = (Z_0 == 0).astype(float) * Z_approximated + Z_0
        rmse = RMSE(Zr, Z_test)
        condition = (i < n_iter) and ((rmse_prev-rmse) > epsilon)
        if condition:
            Z_best = Zr
        if verbose:
            print(f'Training RMSE after {i+1}th iteration equals = {rmse}')
        i += 1
        rmse_prev = rmse
    print(
        f"Returning last, best approximation with RMSE = {RMSE(Z_best, Z_test)}")
    return Z_best


def sgd(Z: np.ndarray, r: int = 8, max_iter: int = 200, alpha: float = 0.05,
        lambd: float = 0.01, batch_size: int = 64) -> np.ndarray:
    """
    Function performs low rank approximation of utility matrix Z via SGD.

    :param Z: Utility matrix Z (users x movies) (may have missing values)
    :type Z: numpy.ndarray
    :param r: Number of components in SGD (ranks)
    :type r: int
    :param max_iter: Maximum number of SGD algorithm iterations
    :type max_iter: int
    :param batch_size: Batch size for computing gradients
    :type batch_size: int
    :return: Rank r matrix - an approximation of Z.
    :rtype: numpy.ndarray
    """
    n, d = Z.shape
    W = np.random.normal(size=(n, r))
    H = np.random.normal(size=(r, d))

    not_nans = np.argwhere(~np.isnan(Z))
    np.random.shuffle(not_nans)

    def loss(W, H, truth):
        s = 0
        for (user_id, movie_id) in not_nans:
            w_i = W[user_id, :]
            h_j = H[:, movie_id]
            s += (w_i.T@h_j-truth[user_id, movie_id])**2 + \
                lambd*((w_i**2).sum()+(h_j**2).sum())
        return s

    def grad(W, H, truth, start):
        DW = np.zeros(shape=(n, r))
        DH = np.zeros(shape=(r, d))
        end = min(start+batch_size, len(not_nans))
        for (user_id, movie_id) in not_nans[start:end]:
            w_i = W[user_id, :]
            h_j = H[:, movie_id]
            t_i_j = truth[user_id, movie_id]
            DW[user_id, :]  += 2*(w_i.T @ h_j-t_i_j)*h_j + lambd*2*w_i
            DH[:, movie_id] += 2*(h_j.T @ w_i-t_i_j)*w_i + lambd*2*h_j
        f = 1/(end-start)
        return DW*f, DH*f

    with tqdm(max_iter) as pbar:
        for i in range(max_iter):
            if (i+1)%(max_iter//4) == 0:
                alpha /= 2
            for start in range(0, len(not_nans), batch_size):
                DW, DH = grad(W, H, Z, start)
                W = W - alpha*DW
                H = H - alpha*DH
            pbar.update()
            if test_ratings is not None:
                pbar.set_postfix(rmse=RMSE(np.dot(W, H), test_ratings))

    Z_approximated = np.dot(W, H)
    return Z_approximated


def fill_zeros(Z):
    """
    Fills missing values in Z with zeros

    :param Z: Utility matrix Z (users x movies)
    :type Z: numpy.ndarray
    :return: matrix Z with NaNs replaced with zeros
    :rtype: numpy.ndarray
    """
    return np.nan_to_num(Z, nan=0.0)


def fill_mean_global(Z):
    """
    Fills missing values in Z with global mean

    :param Z: Utility matrix Z (users x movies)
    :type Z: numpy.ndarray
    :return: matrix Z with NaNs replaced with global mean
    :rtype: numpy.ndarray
    """
    m = np.nanmean(Z)
    return np.nan_to_num(Z, nan=m)


def fill_mean_movies(Z):
    """
    Fills missing values in Z with means per movie

    :param Z: Utility matrix Z (users x movies)
    :type Z: numpy.ndarray
    :return: matrix Z with NaNs replaced with means per movie
    :rtype: numpy.ndarray
    """
    col_mean = np.nanmean(Z, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    inds = np.where(np.isnan(Z))
    Z_copy = Z.copy()
    Z_copy[inds] = np.take(col_mean, inds[1])
    return Z_copy


def fill_mean_users(Z):
    """
    Fills missing values in Z with means per user

    :param Z: Utility matrix Z (users x movies)
    :type Z: numpy.ndarray
    :return: matrix Z with NaNs replaced with means per user
    :rtype: numpy.ndarray
    """
    row_mean = np.nanmean(Z, axis=1)
    row_mean = np.nan_to_num(row_mean, nan=0.0)
    inds = np.where(np.isnan(Z))
    Z_copy = Z.copy()
    Z_copy[inds] = np.take(row_mean, inds[0])
    return Z_copy


def fill_mean_weighted(Z, alpha=0.23):
    """
    Fills missing values in Z with weighted mean between users-mean and movies-mean

    :param Z: Utility matrix Z (users x movies)
    :type Z: numpy.ndarray
    :param alpha: mixing factor
    :type alpha: float
    :return: matrix Z with NaNs replaced with weighted means
    :rtype: numpy.ndarray
    """
    Z_mean_users = fill_mean_users(Z)
    Z_mean_movies = fill_mean_movies(Z)
    return alpha*Z_mean_movies+(1-alpha)*Z_mean_users


best_alpha = 0.23

if __name__ == '__main__':
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

    # global train_ratings, test_ratings
    train_ratings, test_ratings = get_matrices(args.train, args.test)

    # filled_ratings = fill_mean_users(train_ratings)
    filled_ratings = fill_mean_weighted(train_ratings, best_alpha)

    if args.alg == 'NMF':
        approximation = nmf(filled_ratings, 6, max_iter=1000)
    elif args.alg == 'SVD1':
        approximation = svd_1(filled_ratings, 6)
    elif args.alg == 'SVD2':
        approximation = svd_2(
            Z_filled=filled_ratings, Z=train_ratings, Z_test=test_ratings,
            r=7, n_iter=100, epsilon=0.0)
    elif args.alg == 'SGD':
        approximation = sgd(train_ratings)

    rmse_test = RMSE(approximation, test_ratings)

    with open(args.result, 'a') as f:
        f.write(f'{args.alg}: {rmse_test:.2f}\n')
