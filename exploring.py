# %%
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from project import RMSE, nmf, svd_1, svd_2, sgd, get_matrices,\
    fill_mean_global, fill_mean_movies, fill_mean_users,\
    fill_mean_weighted, fill_zeros
train_ratings, test_ratings = get_matrices('train_ratings.csv', 'test_ratings.csv')
# %%
# import matplotlib.pyplot as plt
# rmses = []
# xs = np.linspace(0, 1, 101)
# for x in xs:
#     rmse = RMSE(fill_mean_weighted(train_ratings, x), test_ratings)
#     print(f'{x}: {rmse}')
#     rmses.append(rmse)
# plt.title('RMSE vs mixing coefficient')
# plt.ylabel('RMSE')
# plt.xlabel('$\\alpha$')
# plt.plot(xs, rmses)
# plt.savefig('alpha_tradeoff.png')

# %%
fillers_dict = dict(
    fill_zeros=fill_zeros,
    fill_mean_global=fill_mean_global,
    fill_mean_movies=fill_mean_movies,
    fill_mean_users=fill_mean_users,
    fill_mean_weighted=fill_mean_weighted
)
algs_dict = dict(
    nmf=nmf,
    svd_1=svd_1,
    svd_2=svd_2,
    sgd=sgd
)
fillers = ['fill_zeros', 'fill_mean_global', 'fill_mean_movies', 'fill_mean_users', 'fill_mean_weighted']
algs = ['nmf', 'svd_1', 'svd_2']
rs = [1, 2, 4, 8, 16, 32]
results = np.zeros(shape=(len(fillers), len(algs), len(rs)))
alphas = [0.004, 0.01, 0.04, 0.1]
lambdas = [0.001, 0.004, 0.01]
results_sgd = np.zeros(shape=(len(fillers), len(rs), len(alphas), len(lambdas)))

from itertools import product

for (filler_num, filler_name), (r_num, r), (alpha_num, alpha), (lambda_num, lambd) in product(*map(enumerate, [fillers, rs, alphas, lambdas])):
    filler = fillers_dict[filler_name]
    Z_filled = filler(train_ratings)
    Z_approx = sgd(Z_filled, r=r, alpha=alpha, lambd=lambd)
    rmse = RMSE(Z_approx, test_ratings)
    print(f'{filler_name}, sgd, r={r}, alpha={alpha}, lambda={lambd} - RMSE: {rmse:.3f}')
    results_sgd[filler_num, r_num, alpha_num, lambda_num] = rmse
np.save('results_sgd.npy', results_sgd)

for (filler_num, filler_name), (alg_num, alg_name), (r_num, r) in product(*map(enumerate, [fillers, algs, rs])):
    filler, alg = fillers_dict[filler_name], algs_dict[alg_name]
    Z_filled = filler(train_ratings)
    if alg_name != 'svd_2':
        Z_approx = alg(Z_filled, r=r)
    else:
        Z_approx = alg(Z_filled, train_ratings, test_ratings, r=r)
    rmse = RMSE(Z_approx, test_ratings)
    print(f'{filler_name}, {alg_name}, r={r} - RMSE: {rmse:.3f}')
    results[filler_num, alg_num, r_num] = rmse
np.save('results.npy', results)

# %%
    
