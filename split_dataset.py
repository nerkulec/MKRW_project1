# %%
import numpy as np
import pandas as pd
# %%
def train_test_split(df):
    train = []
    test = []
    for userId in set(df.userId):
        user_ratings = df[df.userId == userId]
        n = len(user_ratings)
        indexes = np.random.choice(n, size=int(n*0.9), replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[indexes] = True # always wear a mask
        train.append(user_ratings[mask])
        test.append(user_ratings[~mask])
    train = pd.concat(train)
    test = pd.concat(test)
    return train, test
# %%
df = pd.read_csv('ml-latest-small/ratings.csv')
train, test = train_test_split(df)
train.to_csv('train_ratings.csv', index=False)
test.to_csv('test_ratings.csv', index=False)
