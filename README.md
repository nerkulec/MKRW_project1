# Methods of classification and Dimensionality reduction. Project #1.

1st project as a part of '_Methods for classification and dimensionality reduction_' course's laboratory @ University of Wrocław.

## Table of contents

- [Introduction](#Introduction)
- [Definition of the project's problem](#Definition-of-the-project-problem)
- [Technologies used](#Technologies-used)
- [Data](#Data)
- [Mathematics behind _algorithms_](#Algorithms)
- [Launch](#Launch)

## Introduction:

We live in times, quite dynamic ones, that brought us rapid, exponential development of new technologies in the broad area of Information Technology, especially Internet. It is impossible to not having heard of a friend or a colleague
buying brand new product based on add-in recommendations they saw while scrolling through their commonly used newsfeed, which is a successful way of e-commerce ability to target us we such an accurate recommendations thanks to great Recommender Systems that are the main topic of this Project. In this project we aim to work on deciphering the magic behind many commonly used techniques in Recommender Systems that are in constant use by such companies like Amazon, Google, Netflix or any local business in your Area and end up using those algorithms on data to perform recommendations. Those recommender systems as it turns out, base on most important mathematical operations we may heard of as eigen-something during our algebra course. Project is divided into two parts which will be described more deeply/briefly in section [Algorithms](#Algorithms).

## Definition of the project problem:

Given set of data consisting of movie ratings of many users, the task is to develop algorithms, train them on 90 per cent of data and perform recommendations (predictions of users' ratings per film) by suggested algorithms i.e NMF, SVD, collaborative SVD and SGD on the remaining 10 % of the data.

## Technologies used:

In this project we will use:

- Python version 3

- Scikit-learn, numpy, pandas, argparse libraries.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install needed libraries according to the example:

```bash
pip install numpy
```

## Data:

Data:
Our data comes from https://grouplens.org/datasets/movielens/ and can be downloaded by clicking here [ml-latest-small.zip](https://grouplens.org/datasets/movielens/ml-latest-small.zip). The main interest of ours is the file 'ratings.cvs' which consists of around 100000 ratings done by around 600 users on around 9000 movies. Its format is as following:

![](https://github.com/nerkulec/MKRW_project1/blob/kuba/images/rating_data.png)

whereas:

- userId is a unique user id,
- MovieId is a unique movie id,
- rating is the rating 0–5 (integer).
- timestamp – to be omitted.

Above presented data is yet not ready to be used by any algorithm, it requires to be:

- firstly split into train & test part in such a way that, predownloaded main file i.e. 'ratings.csv' gets divided into two files 'train_ratings.csv' (train set) and 'test_ratings.csv' (test set) randomly, so that training_ratings.csv contains around 90% of ratings of each user and test_ratings.csv contains the remaining ones.
  (In order to perform train/test split before running main program check [launching](#Launch) instructions.)

- secondly converted into manageable format of a so-called utility 2D matrix (from now on, we will call it **_Z_**) described below, which then is ready to be used by an algorithm:
  Let **_Z_** be a matrix containing training ratings – a matrix of size n × d, where n is the number of users and d is the number of movies.
  Thus, the presented above fragment of ratings.csv – assuming it is all in a training set – is converted to:
  **_Z_** [0,0] = 4.0,
  **_Z_** [0,2] = 4.0,
  **_Z_** [0,5] = 4.0,
  **_Z_** [0,46] = 5.0,
  **_Z_** [0,49] = 5.0, ... .

Of course Z is sparse – many entries of **_Z_** are not defined (either there is no such pair at all, or it is in the test set). This is due to the fact that not all users rated all movies.

## Algorithms:

Our project is divided into two parts, in the first we try to develop algorithms for rating recommendations using such methods like Non-negative matrix factorization (NMF), Singular Value Decomposition(SVD in two approaches) and in the second part we will focus on approximation of the users' rating by finding the minimum of our objective function with Stochastic Gradient Descent.

1. NMF, SVD1, SVD2

2. SGD

## Launch:

How to launch the project?
Thanks to convenient `argparse` library:

- splitting the data can be done as follows:

```bash
python3 split_train_test.py
--original_file file_to_split
--train_file train_file_path
--test_file test_file_path
```

where:

• `file_to_split` is a path to raw file with movie ratings.

• `train_file_path` is a train_rating.csv file's savepath.

• `test_file_path` is a test_rating.csv file's savepath.


- Main program is program is to be called in the Computer's terminal (current directory: repository folder) as in the following example:

```bash

python3 recom_system_IndexNr1_IndexNr2.py
--train train_file_path
--test test_file_path
--alg ALG
--result result_file_path
```

where:

• `train_file_path` is a path to file with the training data (train_ratings.csv)

• `test_file_path` is a path to file with the test data (test_ratings.csv)

• `ALG` is one of the algorithms NMF, SVD1, SVD1, SGD (note – uppercase)

• `result_file_path` is a path to file where a final score will be saved (only this number will appear in this file)


Final score is the metric of algorithm performance (Quality of the recommender system using selected algorithm in `ALG`). In our project the metric of performance is RMSE described below:

Recall the original sparse matrix **_Z_** that can be approximated by algorithms `ALG` in a way described in section [Algorithms](#Algorithms). Approximation of ratings results in a new matrix: **_Z'_**, now the quality of those approximations will be computed with RMSE metric by comparing with sparse test matrix **_T_**, which is in a same format as original training matrix **_Z_** (many entries of **_T_** are not defined (either there is no such pair at all, or it is in the train set). This is due to the fact that not all users rated all movies). In other words **_Z_** comes from the file train_ratings.csv, whereas from the file test_ratings.csv we analogously create the matrix **_T_**.

Now, let <img src="https://github.com/nerkulec/MKRW_project1/blob/kuba/images/tau.png" width="20" heigth="20"/> denote a set of user, movie pairs (u, m) - with ratings present in a test set – then the existing rating is given by **_T_**[u, m]. Assume that your algorithm after training on **_Z_** computes **_Z'_**, a matrix containing elements **_Z'_**[u, m] for (u, m) ∈ <img src="https://github.com/nerkulec/MKRW_project1/blob/kuba/images/tau.png" width="20" heigth="20"/>. Then the quality is computed as root-mean square error:

![](https://github.com/nerkulec/MKRW_project1/blob/kuba/images/rmse.png)
