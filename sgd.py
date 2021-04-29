import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from numba import jit, njit

from project import get_matrices, fill_mean_weighted, RMSE
train_ratings, test_ratings = get_matrices('train_ratings.csv', 'test_ratings.csv')

def sgd(Z: np.ndarray, r: int = 4, max_iter: int = 200, alpha: float = 0.4,
        lambd: float = 0.05, batch_size: int = 512) -> np.ndarray:
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
    H = np.random.normal(size=(d, r))

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

    @njit(fastmath=True)
    def grad(W, H, truth, alpha, batch_size):
        for start in range(0, len(not_nans), batch_size):
            end = min(start+batch_size, len(not_nans))
            
            f = alpha*2/(end-start)
            for i in range(start, end):
                user_id, movie_id = not_nans[i]
                w_i = W[user_id]
                h_j = H[movie_id]
                t_i_j = truth[user_id, movie_id]
                W[user_id]  -= f*((w_i.T @ h_j - t_i_j)*h_j + lambd*w_i)
                H[movie_id] -= f*((h_j.T @ w_i - t_i_j)*w_i + lambd*h_j)

    with tqdm(max_iter) as pbar:
        for i in range(max_iter):
            if (i+1)%(max_iter//4) == 0:
                # alpha *= 0.7
                np.random.shuffle(not_nans)
            grad(W, H, Z, alpha, batch_size)
            pbar.update()
            if test_ratings is not None:
                pbar.set_postfix(rmse=RMSE(np.dot(W, H.T), test_ratings))

    Z_approximated = np.dot(W, H.T)
    return Z_approximated


import argparse
parser = argparse.ArgumentParser(description='searching for optimal hyperparameters for SGD')
parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--r', type=int, default=2)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--lambd', type=float, default=0.01)
args = parser.parse_args()

Z_filled = fill_mean_weighted(train_ratings)
Z_approx = sgd(Z_filled, args.r, args.max_iter, args.alpha, args.lambd, args.batch_size)
rmse = RMSE(Z_approx, test_ratings)

with open('sgd_res.txt', 'a') as f:
    s = f'{args.r}${args.alpha}${args.lambd}${rmse}'
    print(s)
    f.write(s)