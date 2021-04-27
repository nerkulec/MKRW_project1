import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from project import RMSE, nmf, svd_1, svd_2, sgd, get_matrices,\
    fill_mean_global, fill_mean_movies, fill_mean_users,\
    fill_mean_weighted, fill_zeros
train_ratings, test_ratings = get_matrices('train_ratings.csv', 'test_ratings.csv')
# %%
import matplotlib.pyplot as plt
rmses = []
xs = np.linspace(0, 1, 101)
for x in xs:
    rmse = RMSE(fill_mean_weighted(train_ratings, x), test_ratings)
    print(f'{x}: {rmse}')
    rmses.append(rmse)
plt.title('RMSE vs mixing coefficient')
plt.ylabel('RMSE')
plt.xlabel('$\\alpha$')
plt.plot(xs, rmses)
plt.savefig('alpha_tradeoff.png')

# %%