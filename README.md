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

Given set of data consisting of movie ratings of many users, the task is to develop algorithms, train them on 90 per cent of data and perform recommendations (predictions of users' ratings per film) by suggested algorithms i.e NMF, SVD, iterative SVD and SGD on the remaining 10 % of the data.

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

Our project is divided into two parts, in the first we try to develop algorithms for rating recommendations using such methods like Non-negative matrix factorization (NMF), Singular Value Decomposition (SVD in two approaches) and in the second part we will focus on approximation of the users' rating by finding the minimum of objective function, being the loss, with Stochastic Gradient Descent.
All of the algorithmic approaches will try to represent our Utility matrix **_Z_**(nxd) (assume n ≥ d) as a product of two matrices s.t. Z = WH <img src="https://render.githubusercontent.com/render/math?math=Z = WH"> (<img src="https://render.githubusercontent.com/render/math?math=W_{nxd}">W(nxd), H(dxd) <img src="https://render.githubusercontent.com/render/math?math=H_{dxd}">) and approximate Z<img src="https://render.githubusercontent.com/render/math?math=Z"> by taking the first r < d columns and rows of <img src="https://render.githubusercontent.com/render/math?math=W"> and <img src="https://render.githubusercontent.com/render/math?math=H"> respectively. Now <img src="https://render.githubusercontent.com/render/math?math=Z \approx Z_r = W_rH_r"> Z ~ Zr = WrHr ( W(nxr), H(rxd) ). The aim is to approximate Z<img src="https://render.githubusercontent.com/render/math?math=Z"> by <img src="https://render.githubusercontent.com/render/math?math=Z_r">Z˜r, so that <img src="https://render.githubusercontent.com/render/math?math= \|Z - Z_r\|"> ||Z − Z˜r|| is small. Intuitively, we can say that we would like to represent our "user-movie rating" matrix Z <img src="https://render.githubusercontent.com/render/math?math=Z">by two other matrices "user-feature" matrix W <img src="https://render.githubusercontent.com/render/math?math=W">, and "feature-movie" matrix H <img src="https://render.githubusercontent.com/render/math?math=H">, where number of features is equal to chosen r for this desired approximation. Those latent features could be understood as any real features describing movies such as for example science fiction, comedy, documentary etc. or user's taste in movies based on their characteristic. Briefly, The latent features/factors here are the characteristics of the items, for example, the genre of the movie. Futhermore, now W <img src="https://render.githubusercontent.com/render/math?math=W">would describe how much particular user likes some features while watching movies and <img src="https://render.githubusercontent.com/render/math?math=H^T"> H^T would describe how much or to what extent is particular title a bit of each feature. Shortly, they describe relationship between users and latent factors as well as the similarity between items and latent factors.

### I part.

In the first part of our project the main difference is that we need to impute missing entries in the utility matrix so that, the algorithms can process the data and return the approximation of our Z. The core of our recommendation system is that given sparse utility matrix <img src="https://render.githubusercontent.com/render/math?math=Z">, after processing we get non-sparse matrix <img src="https://render.githubusercontent.com/render/math?math=\hat{Z}"> that contains approximated ratings for each user-movie pairs.

Algorithms:

- Singular Value decomposition (SVD1):

Singular value decomposition takes a rectangular matrix <img src="https://render.githubusercontent.com/render/math?math=Z_{nxd}"> of data, in which the <img src="https://render.githubusercontent.com/render/math?math=n"> rows represents the users, and the <img src="https://render.githubusercontent.com/render/math?math=d"> columns represents the movies. The SVD theorem states:
<img src="https://render.githubusercontent.com/render/math?math=Z = U \Lambda^(\frac{1}{2}) V^T = U \Sigma V^T">
Z = U Λ^(1/2) VT = U Σ VT, where:

- U <img src="https://render.githubusercontent.com/render/math?math=U"> is a n x d orthogonal left singular matrix, which represents the relationship between users and latent factors,
- Σ <img src="https://render.githubusercontent.com/render/math?math=\Sigma">is a d x d diagonal matrix, which describes the strength of each latent factor and
- V <img src="https://render.githubusercontent.com/render/math?math=V"> is a d x d orthogonal right singular matrix, which indicates the similarity between items and latent factors.

Calculating the SVD consists of finding the eigenvalues and eigenvectors of ZZT <img src="https://render.githubusercontent.com/render/math?math=ZZ^T"> and ZTZ <img src="https://render.githubusercontent.com/render/math?math=Z^TZ">. The eigenvectors of ZTZ <img src="https://render.githubusercontent.com/render/math?math=Z^TZ"> make up the columns of V <img src="https://render.githubusercontent.com/render/math?math=U">, the eigenvectors of ZZT <img src="https://render.githubusercontent.com/render/math?math=ZZ^T"> make up the columns of <img src="https://render.githubusercontent.com/render/math?math=U">. Also, the singular values in dagonal <img src="https://render.githubusercontent.com/render/math?math=\Sigma"> are square roots of eigenvalues (σ^2 = λ) <img src="https://render.githubusercontent.com/render/math?math=\sigma^2 = \lambda"> from ZZT <img src="https://render.githubusercontent.com/render/math?math=ZZ^T"> or ZTZ <img src="https://render.githubusercontent.com/render/math?math=Z^TZ">. The singular values <img src="https://render.githubusercontent.com/render/math?math=\sigma"> are the diagonal entries of the <img src="https://render.githubusercontent.com/render/math?math=\Sigma">matrix and are arranged in descending order. The singular values are always real numbers. If the matrix Z <img src="https://render.githubusercontent.com/render/math?math=Z"> is a real matrix, then U<img src="https://render.githubusercontent.com/render/math?math=U"> and V <img src="https://render.githubusercontent.com/render/math?math=V"> are also real. Those eigenvectors and their coresponding eigenvalues are the principal componanents of our <img src="https://render.githubusercontent.com/render/math?math=Z"> matrix. Those principal components are in fact our latent features which happen to describe the user-feature and feature-movie relations. Singular values are placed in Σ <img src="https://render.githubusercontent.com/render/math?math=\Sigma"> on diagnals in descending order, thus picking the first top singular value and their corresponding eigenvector would result in collecting the principal compnent/latent feature being the carrier of the most information, which means that the 1st feature best descibes the desired user-feature, item-feature relations. Futhermore, the smaller the singular value gets the less infomation its eigenvector carries and their importance weakens. The next step of SVD would be to reduce number of original latent factors (d -> r) through truncation of our matrices (selecting up to r columns), which as a result is the desired approximation, in fact low-rank-approximation. For now, say rank(Z) <img src="https://render.githubusercontent.com/render/math?math=rank(Z)"> is the number of linearly independent columns in Z<img src="https://render.githubusercontent.com/render/math?math=Z">, thus by truncation of columns in all three matrices (d -> r) from now on we use less principal components/features to describe the user-feature, item-feature releationships. On the other hand we picked those being definitively the most informative for our recommendation system. In order to obtain desidred W and H matrices we can say that <img src="https://render.githubusercontent.com/render/math?math=Z \approx U_r\Sigma_rV_r^T =: U_rH_r ">
Z ≈ UrΣrVTr =: UrHr, depicted below:

<!-- ![](https://github.com/nerkulec/MKRW_project1/blob/kuba/images/rmse.png) -->

In the context of the recommender system, the SVD is used as a collaborative filtering technique.

- Iterative singular value decomposition (SVD2):

- Non-negative Matrix Factorization:

### II part.

- SGD

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

Recall the original sparse matrix **_Z_** that can be approximated by algorithms `ALG` in a way described in section [Algorithms](#Algorithms). Approximation of ratings results in a new matrix: **_Z'_**, now the quality of those approximations will be computed with RMSE metric by comparing with sparse test matrix **_T_**, which is in a same format as original training matrix **_Z_** (many entries of **_T_** are not defined (either there is no such pair at all, or it is in the train set). This is due to the fact that not all users rated all movies). In other words **_Z_** comes from the file train*ratings.csv, whereas from the file test_ratings.csv we analogously create the matrix \*\*\_T*\*\*.

Now, let <img src="https://render.githubusercontent.com/render/math?math=\Tau"> denote a set of user, movie pairs (u, m) - with ratings present in a test set – then the existing rating is given by **_T_**[u, m]. Assume that your algorithm after training on **_Z_** computes **_Z'_**, a matrix containing elements **_Z'_**[u, m] for <img src="https://render.githubusercontent.com/render/math?math=(u, m) \in \Tau">. Then the quality is computed as root-mean square error:

<img src="https://render.githubusercontent.com/render/math?math=RMSE = \displaystyle\sqrt{\frac1{|\Tau|} \sum_{(u, m) \in \Tau} \left(Z'[u, m] - T[u, m] \right)^2}">
